"""Linear mooring matrix models and reduction helpers.

The first mooring layer follows the WEC-Sim Mooring Matrix convention:

    F_moor = F0 - K_moor * q - C_moor * qdot

where ``K_moor`` and ``C_moor`` are 6-DOF linearized stiffness and damping
matrices and ``F0`` is the static pretension/restoring force vector.  The RODM
model usually retains the first five structural DOFs per node, so this module
also provides projection helpers from natural retained global DOFs to SEREP
master coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


DOF_LABELS_6 = ("surge", "sway", "heave", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class LinearMooringMatrix:
    """WEC-Sim-style linear mooring matrix for one 6-DOF attachment point."""

    stiffness: np.ndarray | None = None
    damping: np.ndarray | None = None
    pretension: np.ndarray | None = None
    dof_count: int = 6
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dof_count <= 0:
            raise ValueError("dof_count must be positive")
        stiffness = _as_square_or_zero(self.stiffness, self.dof_count, "stiffness")
        damping = _as_square_or_zero(self.damping, self.dof_count, "damping")
        pretension = _as_vector_or_zero(self.pretension, self.dof_count, "pretension")
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "pretension", pretension)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def force(
        self,
        displacement: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate ``F0 - K q - C qdot`` for one or more 6-DOF states."""

        q = np.asarray(displacement, dtype=float)
        if q.shape[-1] != self.dof_count:
            raise ValueError("displacement last axis must match dof_count")
        if velocity is None:
            qdot = np.zeros_like(q, dtype=float)
        else:
            qdot = np.asarray(velocity, dtype=float)
            if qdot.shape != q.shape:
                raise ValueError("velocity must have the same shape as displacement")
        elastic = np.einsum("ij,...j->...i", self.stiffness, q)
        damping = np.einsum("ij,...j->...i", self.damping, qdot)
        return self.pretension - elastic - damping

    def retained(self, retained_full_dofs_zero_based: Sequence[int]) -> "LinearMooringMatrix":
        """Return a matrix restricted to the retained full-DOF indices."""

        retained = _validate_dof_indices(retained_full_dofs_zero_based, self.dof_count)
        retained_array = np.asarray(retained, dtype=int)
        return LinearMooringMatrix(
            stiffness=self.stiffness[np.ix_(retained, retained)],
            damping=self.damping[np.ix_(retained, retained)],
            pretension=self.pretension[retained_array],
            dof_count=len(retained),
            metadata=self.metadata,
        )

    def is_zero(self) -> bool:
        """Return true when all three linear mooring terms are zero."""

        return not (
            np.any(self.stiffness)
            or np.any(self.damping)
            or np.any(self.pretension)
        )


@dataclass(frozen=True)
class NodalMooringAttachment:
    """Linear mooring attached to one one-based structural node."""

    node_one_based: int
    matrix: LinearMooringMatrix
    name: str = ""

    def __post_init__(self) -> None:
        if self.node_one_based <= 0:
            raise ValueError("node_one_based must be positive")


@dataclass(frozen=True)
class GlobalMooringTerms:
    """Mooring terms in natural retained global DOF order."""

    stiffness: np.ndarray
    damping: np.ndarray
    pretension: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stiffness = _as_square_matrix(self.stiffness, "stiffness")
        damping = _as_square_matrix(self.damping, "damping")
        if stiffness.shape != damping.shape:
            raise ValueError("stiffness and damping must have the same shape")
        pretension = np.asarray(self.pretension, dtype=float).reshape(-1)
        if pretension.size != stiffness.shape[0]:
            raise ValueError("pretension length must match matrix dimensions")
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "pretension", pretension)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ReducedMooringTerms:
    """Mooring terms in reduced/master DOF coordinates."""

    stiffness: np.ndarray
    damping: np.ndarray
    pretension: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stiffness = _as_square_matrix(self.stiffness, "stiffness")
        damping = _as_square_matrix(self.damping, "damping")
        if stiffness.shape != damping.shape:
            raise ValueError("stiffness and damping must have the same shape")
        pretension = np.asarray(self.pretension, dtype=float).reshape(-1)
        if pretension.size != stiffness.shape[0]:
            raise ValueError("pretension length must match matrix dimensions")
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "pretension", pretension)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        """Return true when at least one reduced mooring term is nonzero."""

        return bool(
            np.any(self.stiffness)
            or np.any(self.damping)
            or np.any(self.pretension)
        )


def assemble_nodal_mooring_terms(
    attachments: Sequence[NodalMooringAttachment],
    *,
    total_nodes: int,
    retained_full_dofs_zero_based: Sequence[int],
) -> GlobalMooringTerms:
    """Assemble nodal linear mooring terms in natural retained global order."""

    if total_nodes <= 0:
        raise ValueError("total_nodes must be positive")
    attachment_list = tuple(attachments)
    retained = _validate_dof_indices(retained_full_dofs_zero_based, 6)
    dofs_per_node = len(retained)
    ndof = total_nodes * dofs_per_node
    stiffness = np.zeros((ndof, ndof), dtype=float)
    damping = np.zeros_like(stiffness)
    pretension = np.zeros(ndof, dtype=float)

    attachment_names: list[str] = []
    for attachment in attachment_list:
        if attachment.node_one_based > total_nodes:
            raise ValueError("mooring attachment node is outside total_nodes")
        local = attachment.matrix.retained(retained)
        start = (attachment.node_one_based - 1) * dofs_per_node
        stop = start + dofs_per_node
        block = slice(start, stop)
        stiffness[block, block] += local.stiffness
        damping[block, block] += local.damping
        pretension[block] += local.pretension
        attachment_names.append(attachment.name or f"node_{attachment.node_one_based}")

    return GlobalMooringTerms(
        stiffness=stiffness,
        damping=damping,
        pretension=pretension,
        metadata={
            "attachment_count": len(attachment_list),
            "attachment_names": tuple(attachment_names),
            "retained_full_dofs_zero_based": tuple(int(value) for value in retained),
            "convention": "F_moor = F0 - K*q - C*qdot",
        },
    )


def project_global_mooring_terms_to_reduced(
    terms: GlobalMooringTerms,
    transformation: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    reverse_master_order: bool = False,
    symmetrize_matrices: bool = True,
) -> ReducedMooringTerms:
    """Project natural-order global mooring terms into SEREP master DOFs.

    ``transformation`` maps master coordinates to the disordered retained
    global vector used by the existing reduction code.  ``master_dofs`` and
    ``slave_dofs`` therefore define the same disordered order used during
    structural reduction.
    """

    transform = np.asarray(transformation, dtype=float)
    if transform.ndim != 2:
        raise ValueError("transformation must be a matrix")
    master = np.asarray(master_dofs, dtype=int).reshape(-1)
    slave = np.asarray(slave_dofs, dtype=int).reshape(-1)
    ndof = terms.stiffness.shape[0]
    if transform.shape[0] != ndof:
        raise ValueError("transformation row count must match global DOFs")
    if transform.shape[1] != master.size:
        raise ValueError("transformation column count must match master DOFs")
    if master.size + slave.size != ndof:
        raise ValueError("master_dofs and slave_dofs must cover all global DOFs")

    ordered_master = master[::-1] if reverse_master_order else master
    disordered_order = np.concatenate([ordered_master, slave])
    stiffness_disordered = terms.stiffness[np.ix_(disordered_order, disordered_order)]
    damping_disordered = terms.damping[np.ix_(disordered_order, disordered_order)]
    pretension_disordered = terms.pretension[disordered_order]

    reduced_stiffness = transform.T @ stiffness_disordered @ transform
    reduced_damping = transform.T @ damping_disordered @ transform
    reduced_pretension = transform.T @ pretension_disordered
    if symmetrize_matrices:
        reduced_stiffness = _symmetrize(reduced_stiffness)
        reduced_damping = _symmetrize(reduced_damping)

    return ReducedMooringTerms(
        stiffness=reduced_stiffness,
        damping=reduced_damping,
        pretension=reduced_pretension,
        metadata={
            **dict(terms.metadata),
            "projection": "T.T * global_terms_disordered * T",
            "reverse_master_order": bool(reverse_master_order),
        },
    )


def build_nodal_mooring_reduced_terms(
    attachments: Sequence[NodalMooringAttachment],
    *,
    total_nodes: int,
    retained_full_dofs_zero_based: Sequence[int],
    transformation: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    reverse_master_order: bool = False,
) -> ReducedMooringTerms:
    """Assemble nodal mooring terms and project them to reduced coordinates."""

    global_terms = assemble_nodal_mooring_terms(
        attachments,
        total_nodes=total_nodes,
        retained_full_dofs_zero_based=retained_full_dofs_zero_based,
    )
    return project_global_mooring_terms_to_reduced(
        global_terms,
        transformation,
        master_dofs,
        slave_dofs,
        reverse_master_order=reverse_master_order,
    )


def _as_square_or_zero(
    matrix: np.ndarray | None,
    dof_count: int,
    name: str,
) -> np.ndarray:
    if matrix is None:
        return np.zeros((dof_count, dof_count), dtype=float)
    values = _as_square_matrix(matrix, name)
    if values.shape != (dof_count, dof_count):
        raise ValueError(f"{name} must have shape ({dof_count}, {dof_count})")
    return values


def _as_vector_or_zero(
    vector: np.ndarray | None,
    dof_count: int,
    name: str,
) -> np.ndarray:
    if vector is None:
        return np.zeros(dof_count, dtype=float)
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != dof_count:
        raise ValueError(f"{name} must have length {dof_count}")
    return values


def _as_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return values


def _validate_dof_indices(values: Sequence[int], dof_count: int) -> tuple[int, ...]:
    indices = tuple(int(value) for value in values)
    if not indices:
        raise ValueError("retained DOF list must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("retained DOF indices must be unique")
    if min(indices) < 0 or max(indices) >= dof_count:
        raise ValueError("retained DOF indices are outside the source DOF count")
    return indices


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
