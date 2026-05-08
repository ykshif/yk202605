"""Frequency-domain connector relative-motion and force recovery.

The RODM hydroelastic response uses retained structural DOFs, commonly five
DOFs per node after removing the local ``rz`` rotation. This module works
directly with that retained global response vector ``x_hat``.

For connector ``j``:

``delta_hat_j = G_j @ x_hat``

``force_hat_j = (K_j + 1j * omega * C_j) @ delta_hat_j``

The same ``G_j`` used to assemble connector stiffness can therefore be reused
to recover connector force, keeping the equation of motion and post-processing
consistent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from offshore_energy_sim.structure.hinges import hinge_coupling_matrix


MatrixLike = Any
DofMap = Mapping[Any, Any] | Callable[[Any, Any], int]
DEFAULT_FULL_DOF_LABELS = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass(frozen=True)
class Connector:
    """One frequency-domain connector described by a relative-motion operator.

    Parameters
    ----------
    cid:
        Stable connector identifier.
    G:
        Sparse relative-motion operator with shape ``(m, ndof)``.
    K:
        Connector stiffness. Accepted forms are scalar, diagonal vector, dense
        square matrix, or sparse square matrix.
    C:
        Optional connector damping in the same accepted forms as ``K``.
    labels:
        Optional names for the ``m`` connector components.
    meta:
        Optional metadata such as node ids, connector type, or local frame name.
    """

    cid: str
    G: sparse.csr_matrix
    K: MatrixLike
    C: MatrixLike | None = None
    labels: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        G = sparse.csr_matrix(self.G)
        if G.ndim != 2:
            raise ValueError("G must be a 2D matrix")

        component_count = G.shape[0]
        labels = tuple(self.labels) if self.labels else tuple(
            f"dof_{index}" for index in range(component_count)
        )
        if len(labels) != component_count:
            raise ValueError("labels length must match the number of rows in G")

        object.__setattr__(self, "G", G)
        object.__setattr__(self, "K", _as_square_csr(self.K, component_count, "K"))
        object.__setattr__(
            self,
            "C",
            None if self.C is None else _as_square_csr(self.C, component_count, "C"),
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def component_count(self) -> int:
        """Number of connector relative-motion components."""

        return self.G.shape[0]

    @property
    def ndof(self) -> int:
        """Number of global response DOFs referenced by this connector."""

        return self.G.shape[1]


def _as_square_csr(value: MatrixLike, size: int, name: str) -> sparse.csr_matrix:
    """Convert scalar/vector/matrix input into a square CSR matrix."""

    if sparse.issparse(value):
        matrix = value.tocsr()
        if matrix.shape != (size, size):
            raise ValueError(f"{name} shape must be {(size, size)}, got {matrix.shape}")
        return matrix

    array = np.asarray(value)
    if array.ndim == 0:
        return (sparse.eye(size, format="csr", dtype=array.dtype) * array.item()).tocsr()
    if array.ndim == 1:
        if array.shape[0] != size:
            raise ValueError(f"{name} diagonal length must be {size}, got {array.shape[0]}")
        return sparse.diags(array, format="csr")
    if array.ndim == 2:
        if array.shape != (size, size):
            raise ValueError(f"{name} shape must be {(size, size)}, got {array.shape}")
        return sparse.csr_matrix(array)
    raise ValueError(f"{name} must be scalar, vector, square matrix, or sparse matrix")


def _validate_dofs(ndof: int, dofs: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate and normalize global DOF indices."""

    normalized = tuple(int(dof) for dof in dofs)
    if any(dof < 0 or dof >= ndof for dof in normalized):
        raise ValueError(f"{name} contains DOF outside [0, {ndof})")
    return normalized


def _component_transform(
    component_count: int,
    R: MatrixLike | None = None,
    select: Sequence[int] | np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Build ``T`` so transformed relative motion is ``T @ delta``.

    ``R`` is interpreted as a global-to-local component transform. If
    ``component_count == 6`` and ``R`` is ``3x3``, it is expanded to a block
    diagonal transform for translations and rotations. ``select`` is applied
    after ``R``.
    """

    if R is None:
        transform = sparse.eye(component_count, format="csr")
    else:
        raw_R = R if sparse.issparse(R) else np.asarray(R)
        if sparse.issparse(raw_R):
            if raw_R.shape == (3, 3) and component_count == 6:
                R_matrix = sparse.block_diag((raw_R, raw_R), format="csr")
            else:
                R_matrix = raw_R.tocsr()
        else:
            if raw_R.ndim != 2:
                raise ValueError("R must be a 2D matrix")
            if raw_R.shape == (3, 3) and component_count == 6:
                R_matrix = sparse.block_diag((raw_R, raw_R), format="csr")
            else:
                R_matrix = sparse.csr_matrix(raw_R)

        if R_matrix.shape[1] != component_count:
            raise ValueError(
                "R column count must match the number of relative-motion components"
            )
        transform = R_matrix

    if select is None:
        return transform.tocsr()

    select_array = np.asarray(select)
    if select_array.dtype == bool:
        if select_array.size != transform.shape[0]:
            raise ValueError("Boolean select length must match transformed component count")
        selected_indices = tuple(int(index) for index in np.flatnonzero(select_array))
    else:
        selected_indices = tuple(int(index) for index in select_array.tolist())

    if any(index < 0 or index >= transform.shape[0] for index in selected_indices):
        raise ValueError("select contains component index outside transformed range")

    rows = np.arange(len(selected_indices))
    cols = np.asarray(selected_indices)
    data = np.ones(len(selected_indices))
    selector = sparse.csr_matrix((data, (rows, cols)), shape=(len(selected_indices), transform.shape[0]))
    return (selector @ transform).tocsr()


def build_direct_relative_G(
    ndof: int,
    dofs_a: Sequence[int],
    dofs_b: Sequence[int],
    R: MatrixLike | None = None,
    select: Sequence[int] | np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Construct ``G`` for direct endpoint relative motion.

    The raw relative motion is ``x[dofs_a] - x[dofs_b]``. ``dofs_a`` and
    ``dofs_b`` may contain 3 components, 5 retained RODM components, 6 full
    rigid-body components, or any other matching component count.
    """

    dofs_a = _validate_dofs(ndof, dofs_a, "dofs_a")
    dofs_b = _validate_dofs(ndof, dofs_b, "dofs_b")
    if len(dofs_a) != len(dofs_b):
        raise ValueError("dofs_a and dofs_b must have the same length")

    component_count = len(dofs_a)
    rows = np.repeat(np.arange(component_count), 2)
    cols = np.asarray([col for pair in zip(dofs_a, dofs_b) for col in pair])
    data = np.asarray([value for _ in range(component_count) for value in (1.0, -1.0)])
    raw = sparse.csr_matrix((data, (rows, cols)), shape=(component_count, ndof))
    transform = _component_transform(component_count, R=R, select=select)
    return (transform @ raw).tocsr()


def _retained_full_dof_indices(
    dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return full local DOF indices retained in the response vector."""

    removed = set(removed_full_dofs_zero_based)
    return tuple(index for index in range(dofs_per_node) if index not in removed)


def _retained_labels(
    *,
    dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int],
    full_dof_labels: Sequence[str],
) -> tuple[str, ...]:
    """Return retained DOF labels matching the response vector order."""

    if len(full_dof_labels) != dofs_per_node:
        raise ValueError("full_dof_labels length must match dofs_per_node")
    keep = _retained_full_dof_indices(dofs_per_node, removed_full_dofs_zero_based)
    return tuple(full_dof_labels[index] for index in keep)


def _retained_hinge_stiffness(
    hinge,
    removed_full_dofs_zero_based: Sequence[int],
) -> sparse.csr_matrix:
    """Return hinge stiffness matrix in retained response DOF order."""

    full_coupling = hinge_coupling_matrix(
        hinge.k_hinge,
        dofs_per_node=hinge.dofs_per_node,
        released_dofs_zero_based=hinge.released_dofs_zero_based,
        released_dof_stiffness=hinge.released_dof_stiffness,
    )
    keep = _retained_full_dof_indices(hinge.dofs_per_node, removed_full_dofs_zero_based)
    return sparse.csr_matrix(full_coupling[np.ix_(keep, keep)])


def _retained_indices_for_full_dofs(
    full_dofs_zero_based: Sequence[int],
    *,
    dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int],
) -> tuple[int, ...]:
    """Map full local DOF indices to retained local DOF indices."""

    keep = _retained_full_dof_indices(dofs_per_node, removed_full_dofs_zero_based)
    retained_index_by_full = {full_index: index for index, full_index in enumerate(keep)}
    return tuple(
        retained_index_by_full[full_index]
        for full_index in full_dofs_zero_based
        if full_index in retained_index_by_full
    )


def build_hinge_pair_connectors(
    hinges,
    *,
    total_nodes: int,
    response_dofs_per_node: int,
    removed_full_dofs_zero_based: Sequence[int] = (),
    full_dof_labels: Sequence[str] = DEFAULT_FULL_DOF_LABELS,
    ndof: int | None = None,
    cid_prefix: str = "hinge",
) -> tuple[Connector, ...]:
    """Build one ``Connector`` per paired hinge node.

    This converts legacy/published hinge specs into the generic ``Connector``
    representation. Each returned connector uses the direct relative operator
    ``G`` for one paired node, so the same ``G`` can be used for both dynamic
    stiffness assembly and force recovery.
    """

    if ndof is None:
        ndof = total_nodes * response_dofs_per_node
    expected_ndof = total_nodes * response_dofs_per_node
    if ndof != expected_ndof:
        raise ValueError(f"ndof must be {expected_ndof} for the provided node/DOF counts")

    connectors: list[Connector] = []
    for hinge_line, hinge in enumerate(hinges, start=1):
        keep = _retained_full_dof_indices(hinge.dofs_per_node, removed_full_dofs_zero_based)
        if len(keep) != response_dofs_per_node:
            raise ValueError(
                "Retained hinge DOFs do not match response_dofs_per_node: "
                f"{len(keep)} vs {response_dofs_per_node}"
            )

        labels = _retained_labels(
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
            full_dof_labels=full_dof_labels,
        )
        stiffness = _retained_hinge_stiffness(hinge, removed_full_dofs_zero_based)
        released_retained_indices = _retained_indices_for_full_dofs(
            hinge.released_dofs_zero_based,
            dofs_per_node=hinge.dofs_per_node,
            removed_full_dofs_zero_based=removed_full_dofs_zero_based,
        )
        hinge_name = getattr(hinge, "name", "") or f"hinge line {hinge_line}"

        for pair_index, (node_a, node_b) in enumerate(hinge.node_pairs_one_based, start=1):
            if not (1 <= node_a <= total_nodes and 1 <= node_b <= total_nodes):
                raise ValueError(f"Hinge node pair is outside response nodes: {(node_a, node_b)}")

            dofs_a = tuple(
                (node_a - 1) * response_dofs_per_node + local_index
                for local_index in range(response_dofs_per_node)
            )
            dofs_b = tuple(
                (node_b - 1) * response_dofs_per_node + local_index
                for local_index in range(response_dofs_per_node)
            )
            G = build_direct_relative_G(ndof, dofs_a, dofs_b)
            cid = f"{cid_prefix}_{hinge_line:03d}_{pair_index:03d}"
            connectors.append(
                Connector(
                    cid=cid,
                    G=G,
                    K=stiffness,
                    labels=labels,
                    meta={
                        "hinge_line": hinge_line,
                        "hinge_name": hinge_name,
                        "pair_index": pair_index,
                        "node_a": node_a,
                        "node_b": node_b,
                        "released_full_dofs_zero_based": tuple(hinge.released_dofs_zero_based),
                        "released_retained_indices": released_retained_indices,
                        "released_labels": tuple(labels[index] for index in released_retained_indices),
                        "k_hinge": hinge.k_hinge,
                        "released_dof_stiffness": hinge.released_dof_stiffness,
                    },
                )
            )

    return tuple(connectors)


def build_case_hinge_pair_connectors(case, *, cid_prefix: str = "hinge") -> tuple[Connector, ...]:
    """Build one ``Connector`` per hinge node pair from a standard case object."""

    total_nodes = getattr(case, "total_nodes", None)
    if total_nodes is None:
        total_nodes = case.grid.total_nodes

    return build_hinge_pair_connectors(
        case.hinges,
        total_nodes=total_nodes,
        response_dofs_per_node=case.retained_dofs_per_node,
        removed_full_dofs_zero_based=case.removed_full_dofs_zero_based,
        cid_prefix=cid_prefix,
    )


def _lookup_dof_index(
    dofmap: DofMap,
    node: Any,
    dof_name: Any,
    local_index: int,
) -> int:
    """Return one global DOF index from a flexible node/DOF map."""

    if callable(dofmap):
        return int(dofmap(node, dof_name))

    if (node, dof_name) in dofmap:
        return int(dofmap[(node, dof_name)])
    if (node, local_index) in dofmap:
        return int(dofmap[(node, local_index)])

    if node not in dofmap:
        raise KeyError(f"Node {node!r} is not present in dofmap")
    node_entry = dofmap[node]

    if isinstance(node_entry, Mapping):
        if dof_name in node_entry:
            return int(node_entry[dof_name])
        if local_index in node_entry:
            return int(node_entry[local_index])
        raise KeyError(f"DOF {dof_name!r} is not present for node {node!r}")

    return int(node_entry[local_index])


def build_weighted_endpoint_operator(
    ndof: int,
    node_weights: Mapping[Any, float] | Sequence[tuple[Any, float]],
    dofmap: DofMap,
    dof_names: Sequence[Any],
) -> sparse.csr_matrix:
    """Build endpoint extraction/interpolation operator ``H``.

    ``H @ x`` returns the weighted virtual-endpoint motion. For each requested
    DOF component, the row is ``sum_i weight_i * x[node_i, dof]``.
    """

    items = tuple(node_weights.items() if isinstance(node_weights, Mapping) else node_weights)
    if not items:
        raise ValueError("node_weights must contain at least one node")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for local_index, dof_name in enumerate(dof_names):
        for node, weight in items:
            col = _lookup_dof_index(dofmap, node, dof_name, local_index)
            if col < 0 or col >= ndof:
                raise ValueError(f"Mapped DOF index {col} is outside [0, {ndof})")
            rows.append(local_index)
            cols.append(col)
            data.append(float(weight))

    return sparse.csr_matrix((data, (rows, cols)), shape=(len(dof_names), ndof))


def build_weighted_relative_G(
    H_a: MatrixLike,
    H_b: MatrixLike,
    R: MatrixLike | None = None,
    select: Sequence[int] | np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Construct ``G = T @ (H_a - H_b)`` for weighted virtual endpoints."""

    H_a = sparse.csr_matrix(H_a)
    H_b = sparse.csr_matrix(H_b)
    if H_a.shape != H_b.shape:
        raise ValueError(f"H_a and H_b must have the same shape, got {H_a.shape} and {H_b.shape}")

    raw = H_a - H_b
    transform = _component_transform(raw.shape[0], R=R, select=select)
    return (transform @ raw).tocsr()


def _dynamic_matrix(connector: Connector, omega: float) -> sparse.csr_matrix:
    """Return ``K + 1j * omega * C`` for one connector."""

    matrix = connector.K.astype(np.complex128)
    if connector.C is not None:
        matrix = matrix + (1j * float(omega)) * connector.C.astype(np.complex128)
    return matrix.tocsr()


def assemble_connector_dynamic_stiffness(
    ndof: int,
    connectors: Sequence[Connector],
    omega: float,
) -> sparse.csr_matrix:
    """Assemble ``Zc = sum(G.T @ (K + i omega C) @ G)``."""

    if np.ndim(omega) != 0:
        raise ValueError("assemble_connector_dynamic_stiffness expects scalar omega")

    Zc = sparse.csr_matrix((ndof, ndof), dtype=np.complex128)
    for connector in connectors:
        if connector.G.shape[1] != ndof:
            raise ValueError(
                f"Connector {connector.cid!r} G has ndof {connector.G.shape[1]}, expected {ndof}"
            )
        dynamic_matrix = _dynamic_matrix(connector, float(omega))
        Zc = Zc + connector.G.T @ dynamic_matrix @ connector.G
    return Zc.tocsr()


def _as_case_matrix(x_hat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(ndof, n_cases)`` matrix and whether input was a vector."""

    array = np.asarray(x_hat, dtype=np.complex128)
    if array.ndim == 1:
        return array.reshape(-1, 1), True
    if array.ndim == 2:
        return array, False
    raise ValueError("x_hat must have shape (ndof,) or (ndof, n_cases)")


def _omega_by_case(omega: float | Sequence[float] | np.ndarray, n_cases: int) -> np.ndarray:
    """Normalize scalar or per-case omega to a 1D array."""

    omega_array = np.asarray(omega, dtype=float)
    if omega_array.ndim == 0:
        return np.full(n_cases, float(omega_array))
    if omega_array.shape == (n_cases,):
        return omega_array
    raise ValueError(f"omega must be scalar or have shape ({n_cases},)")


def recover_connector_response(
    x_hat: np.ndarray,
    omega: float | Sequence[float] | np.ndarray,
    connectors: Sequence[Connector],
) -> dict[str, dict[str, np.ndarray | Connector]]:
    """Recover complex relative motion and force for all connectors.

    For vector input ``x_hat.shape == (ndof,)``, each output ``delta_hat`` and
    ``force_hat`` has shape ``(m,)``. For batched input
    ``x_hat.shape == (ndof, n_cases)``, outputs have shape ``(m, n_cases)``.
    """

    X, squeeze = _as_case_matrix(x_hat)
    ndof, n_cases = X.shape
    omega_values = _omega_by_case(omega, n_cases)
    recovered: dict[str, dict[str, np.ndarray | Connector]] = {}

    for connector in connectors:
        if connector.G.shape[1] != ndof:
            raise ValueError(
                f"Connector {connector.cid!r} G has ndof {connector.G.shape[1]}, expected {ndof}"
            )
        delta_hat = connector.G @ X
        force_hat = np.empty_like(delta_hat, dtype=np.complex128)
        for case_index, omega_value in enumerate(omega_values):
            force_hat[:, case_index] = _dynamic_matrix(connector, omega_value) @ delta_hat[
                :, case_index
            ]

        if squeeze:
            recovered[connector.cid] = {
                "connector": connector,
                "delta_hat": np.asarray(delta_hat[:, 0]).reshape(-1),
                "force_hat": np.asarray(force_hat[:, 0]).reshape(-1),
            }
        else:
            recovered[connector.cid] = {
                "connector": connector,
                "delta_hat": np.asarray(delta_hat),
                "force_hat": np.asarray(force_hat),
            }

    return recovered


def _weighted_inner(a: np.ndarray, b: np.ndarray, weights: MatrixLike | None) -> float:
    """Return weighted real inner product ``a.T @ W @ b``."""

    if weights is None:
        return float(a @ b)
    if sparse.issparse(weights):
        return float(a @ (weights @ b))

    weight_array = np.asarray(weights)
    if weight_array.ndim == 1:
        if weight_array.shape[0] != a.shape[0]:
            raise ValueError("weights vector length must match z_hat component count")
        return float((weight_array * a) @ b)
    if weight_array.ndim == 2:
        if weight_array.shape != (a.shape[0], a.shape[0]):
            raise ValueError("weights matrix shape must match z_hat component count")
        return float(a @ weight_array @ b)
    raise ValueError("weights must be None, vector, square matrix, or sparse matrix")


def harmonic_vector_norm_envelope(
    z_hat: np.ndarray,
    weights: MatrixLike | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return ``max_phi ||real(z_hat * exp(i phi))||`` and controlling angle.

    ``z_hat`` may be a vector ``(m,)`` or a batch matrix ``(m, n_cases)``.
    The returned angle is in radians and uses the same harmonic convention.
    """

    Z, squeeze = _as_case_matrix(z_hat)
    envelopes = np.empty(Z.shape[1], dtype=float)
    angles = np.empty(Z.shape[1], dtype=float)

    for case_index in range(Z.shape[1]):
        a = Z[:, case_index].real
        b = Z[:, case_index].imag
        A = _weighted_inner(a, a, weights)
        D = _weighted_inner(b, b, weights)
        B = _weighted_inner(a, b, weights)
        harmonic_matrix = np.array([[A, -B], [-B, D]], dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(harmonic_matrix)
        controlling_index = int(np.argmax(eigenvalues))
        max_value = max(float(eigenvalues[controlling_index]), 0.0)
        if max_value == 0.0:
            envelopes[case_index] = 0.0
            angles[case_index] = 0.0
            continue
        vector = eigenvectors[:, controlling_index]
        envelopes[case_index] = float(np.sqrt(max_value))
        angles[case_index] = float(np.arctan2(vector[1], vector[0]))

    if squeeze:
        return float(envelopes[0]), float(angles[0])
    return envelopes, angles


def connector_force_envelope(
    force_hat,
    weights: MatrixLike | Mapping[str, MatrixLike] | None = None,
):
    """Compute harmonic force envelopes for connectors and cases.

    ``force_hat`` can be the dictionary returned by
    :func:`recover_connector_response`, a mapping from connector id to force
    arrays, or one force array.
    """

    if isinstance(force_hat, Mapping):
        output = {}
        for cid, item in force_hat.items():
            if isinstance(item, Mapping) and "force_hat" in item:
                values = item["force_hat"]
            else:
                values = item
            connector_weights = weights.get(cid) if isinstance(weights, Mapping) else weights
            envelope, angle = harmonic_vector_norm_envelope(values, weights=connector_weights)
            output[cid] = {
                "envelope": envelope,
                "controlling_angle": angle,
            }
        return output

    return harmonic_vector_norm_envelope(force_hat, weights=weights)
