"""Connector internal-force recovery for hinge/interconnection studies.

For one connector between nodes ``a`` and ``b`` the local generalized
displacement jump is

``delta_u = u_a - u_b``.

With a linear connector stiffness ``K_c``, the generalized connector action
on side ``a`` is

``q_a = K_c @ delta_u``, and ``q_b = -q_a``.

The translational entries of ``q`` are connector forces, while rotational
entries are connector moments. An ideal hinge is represented by zero stiffness
in the released rotational DOF, so that released moment component is zero. An
elastic hinge uses a nonzero released rotational stiffness, producing a moment
proportional to the relative rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from offshore_energy_sim.structure.hinges import hinge_coupling_matrix


DEFAULT_FULL_DOF_LABELS = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass(frozen=True)
class ConnectorForceResult:
    """Recovered force/moment for one paired connector node.

    The displacement and force vectors use the retained DOF order after any
    full-model DOFs have been removed. For the standard RODM hydroelastic
    workflow this is usually ``[ux, uy, uz, rx, ry]``.
    """

    hinge_line: int
    hinge_name: str
    pair_index: int
    node_a_one_based: int
    node_b_one_based: int
    retained_dof_labels: tuple[str, ...]
    relative_displacement: np.ndarray
    generalized_force: np.ndarray
    stiffness_diagonal: np.ndarray
    shear_force_abs: float
    bending_moment_abs: float
    released_moment_abs: float
    max_component_abs: float

    def as_summary_dict(self) -> dict[str, object]:
        """Return a CSV/JSON-friendly summary without complex vectors."""

        return {
            "hinge_line": self.hinge_line,
            "hinge_name": self.hinge_name,
            "pair_index": self.pair_index,
            "node_a": self.node_a_one_based,
            "node_b": self.node_b_one_based,
            "shear_force_abs": self.shear_force_abs,
            "bending_moment_abs": self.bending_moment_abs,
            "released_moment_abs": self.released_moment_abs,
            "max_component_abs": self.max_component_abs,
        }


def retained_full_dof_indices(
    dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return full-DOF indices retained in the response vector."""

    removed = set(removed_full_dofs_zero_based)
    return tuple(index for index in range(dofs_per_node) if index not in removed)


def retained_dof_labels(
    *,
    dofs_per_node: int = 6,
    removed_full_dofs_zero_based: Sequence[int] = (),
    full_dof_labels: Sequence[str] = DEFAULT_FULL_DOF_LABELS,
) -> tuple[str, ...]:
    """Return retained DOF labels after full-model DOF removal."""

    if len(full_dof_labels) != dofs_per_node:
        raise ValueError("full_dof_labels length must match dofs_per_node")
    keep = retained_full_dof_indices(dofs_per_node, removed_full_dofs_zero_based)
    return tuple(full_dof_labels[index] for index in keep)


def retained_hinge_coupling_matrix(
    hinge,
    removed_full_dofs_zero_based: Sequence[int] = (),
) -> np.ndarray:
    """Return a hinge coupling matrix matching the retained response DOFs."""

    full_coupling = hinge_coupling_matrix(
        hinge.k_hinge,
        dofs_per_node=hinge.dofs_per_node,
        released_dofs_zero_based=hinge.released_dofs_zero_based,
        released_dof_stiffness=hinge.released_dof_stiffness,
    )
    keep = retained_full_dof_indices(hinge.dofs_per_node, removed_full_dofs_zero_based)
    return full_coupling[np.ix_(keep, keep)]


def _map_full_dofs_to_retained(
    full_dofs_zero_based: Sequence[int],
    *,
    dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int],
) -> tuple[int, ...]:
    """Map full-model local DOF indices to retained-response local indices."""

    keep = retained_full_dof_indices(dofs_per_node, removed_full_dofs_zero_based)
    retained_index_by_full = {full_index: index for index, full_index in enumerate(keep)}
    return tuple(
        retained_index_by_full[full_index]
        for full_index in full_dofs_zero_based
        if full_index in retained_index_by_full
    )


def _vector_norm_abs(values: np.ndarray) -> float:
    """Return Euclidean norm of complex/vector magnitudes."""

    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(np.abs(values)))


def compute_hinge_connector_forces(
    response: np.ndarray,
    hinges,
    *,
    total_nodes: int,
    response_dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int] = (),
    shear_full_dofs_zero_based: Sequence[int] = (2,),
    bending_moment_full_dofs_zero_based: Sequence[int] = (3, 4, 5),
    full_dof_labels: Sequence[str] = DEFAULT_FULL_DOF_LABELS,
) -> list[ConnectorForceResult]:
    """Recover connector shear forces and bending moments from a response.

    Parameters
    ----------
    response:
        Global complex displacement response. Shape can be flat or
        ``(total_nodes * response_dofs_per_node, 1)``.
    hinges:
        Iterable of hinge specs exposing ``node_pairs_one_based``,
        ``k_hinge``, ``dofs_per_node``, ``released_dofs_zero_based``, and
        ``released_dof_stiffness``.
    total_nodes:
        Number of structural nodes in the response vector.
    response_dofs_per_node:
        Number of local DOFs retained in ``response``.
    removed_full_dofs_zero_based:
        Full-model local DOFs removed before forming ``response``. For the
        current 10x10 workflow this is usually ``(5,)``.
    shear_full_dofs_zero_based:
        Full-model translational DOFs used for the reported shear resultant.
        The default reports vertical shear ``Fz``.
    bending_moment_full_dofs_zero_based:
        Full-model rotational DOFs used for the reported bending resultant.
        Removed DOFs are ignored automatically.

    Returns
    -------
    list[ConnectorForceResult]
        One item for each paired connector node. ``generalized_force`` is the
        action on side ``a``; the action on side ``b`` is its negative.
    """

    nodal_response = np.asarray(response).reshape(total_nodes, response_dofs_per_node)
    rows: list[ConnectorForceResult] = []

    for hinge_line, hinge in enumerate(hinges, start=1):
        coupling = retained_hinge_coupling_matrix(
            hinge,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
        )
        if coupling.shape != (response_dofs_per_node, response_dofs_per_node):
            raise ValueError(
                "Retained hinge coupling shape does not match response DOFs: "
                f"{coupling.shape} vs {(response_dofs_per_node, response_dofs_per_node)}"
            )

        shear_indices = _map_full_dofs_to_retained(
            shear_full_dofs_zero_based,
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
        )
        bending_indices = _map_full_dofs_to_retained(
            bending_moment_full_dofs_zero_based,
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
        )
        released_moment_indices = _map_full_dofs_to_retained(
            [
                dof
                for dof in hinge.released_dofs_zero_based
                if dof in bending_moment_full_dofs_zero_based
            ],
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
        )
        labels = retained_dof_labels(
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
            full_dof_labels=full_dof_labels,
        )

        for pair_index, (node_a, node_b) in enumerate(hinge.node_pairs_one_based, start=1):
            if not (1 <= node_a <= total_nodes and 1 <= node_b <= total_nodes):
                raise ValueError(f"Hinge node pair is outside response nodes: {(node_a, node_b)}")

            relative_displacement = nodal_response[node_a - 1] - nodal_response[node_b - 1]
            generalized_force = coupling @ relative_displacement
            hinge_name = getattr(hinge, "name", "") or f"hinge line {hinge_line}"

            rows.append(
                ConnectorForceResult(
                    hinge_line=hinge_line,
                    hinge_name=hinge_name,
                    pair_index=pair_index,
                    node_a_one_based=node_a,
                    node_b_one_based=node_b,
                    retained_dof_labels=labels,
                    relative_displacement=relative_displacement,
                    generalized_force=generalized_force,
                    stiffness_diagonal=np.diag(coupling),
                    shear_force_abs=_vector_norm_abs(generalized_force[list(shear_indices)]),
                    bending_moment_abs=_vector_norm_abs(generalized_force[list(bending_indices)]),
                    released_moment_abs=_vector_norm_abs(
                        generalized_force[list(released_moment_indices)]
                    ),
                    max_component_abs=float(np.abs(generalized_force).max()),
                )
            )

    return rows


def compute_case_hinge_connector_forces(
    case,
    response: np.ndarray,
    *,
    shear_full_dofs_zero_based: Sequence[int] = (2,),
    bending_moment_full_dofs_zero_based: Sequence[int] = (3, 4, 5),
) -> list[ConnectorForceResult]:
    """Recover connector forces for a standard case object with hinges.

    This convenience wrapper matches ``ComplexHingeCase`` and similar case
    objects that expose ``hinges``, ``removed_full_dofs_zero_based``, and
    ``retained_dofs_per_node``.
    """

    total_nodes = getattr(case, "total_nodes", None)
    if total_nodes is None:
        total_nodes = case.grid.total_nodes

    return compute_hinge_connector_forces(
        response,
        case.hinges,
        total_nodes=total_nodes,
        response_dofs_per_node=case.retained_dofs_per_node,
        removed_full_dofs_zero_based=case.removed_full_dofs_zero_based,
        shear_full_dofs_zero_based=shear_full_dofs_zero_based,
        bending_moment_full_dofs_zero_based=bending_moment_full_dofs_zero_based,
    )


def connector_force_results_to_rows(
    results: Sequence[ConnectorForceResult],
) -> list[dict[str, object]]:
    """Convert connector-force results to CSV/JSON-friendly dictionaries."""

    return [result.as_summary_dict() for result in results]
