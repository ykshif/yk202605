"""Modal reduction helpers preserving the current SEREP formulas."""

from __future__ import annotations

import numpy as np


def transform_mass_matrix(consistent_mass_matrix: np.ndarray, beta: float) -> np.ndarray:
    """Blend consistent and lumped mass matrices.

    `beta=0` returns the original consistent mass matrix, matching the current
    paper-reproduction scripts. `beta=1` returns a row-sum lumped matrix.
    """

    lumped_mass_matrix = np.diag(consistent_mass_matrix.sum(axis=1))
    return beta * lumped_mass_matrix + (1 - beta) * consistent_mass_matrix


def _reorder_master_slave_blocks(
    stiffness: np.ndarray,
    mass: np.ndarray,
    slave_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder matrices into `[master, slave]` block order."""

    slave_dofs = np.sort(slave_dofs)
    master_dofs = np.setdiff1d(np.arange(stiffness.shape[0]), slave_dofs)
    return _reorder_dof_blocks(stiffness, mass, master_dofs, slave_dofs)


def _reorder_dof_blocks(
    stiffness: np.ndarray,
    mass: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder matrices into an explicit `[master, slave]` block order."""

    mass_mm = mass[master_dofs[:, np.newaxis], master_dofs]
    mass_ms = mass[master_dofs[:, np.newaxis], slave_dofs]
    mass_sm = mass[slave_dofs[:, np.newaxis], master_dofs]
    mass_ss = mass[slave_dofs[:, np.newaxis], slave_dofs]

    stiff_mm = stiffness[master_dofs[:, np.newaxis], master_dofs]
    stiff_ms = stiffness[master_dofs[:, np.newaxis], slave_dofs]
    stiff_sm = stiffness[slave_dofs[:, np.newaxis], master_dofs]
    stiff_ss = stiffness[slave_dofs[:, np.newaxis], slave_dofs]

    reordered_mass = np.vstack([np.hstack([mass_mm, mass_ms]), np.hstack([mass_sm, mass_ss])])
    reordered_stiffness = np.vstack(
        [np.hstack([stiff_mm, stiff_ms]), np.hstack([stiff_sm, stiff_ss])]
    )
    return reordered_stiffness, reordered_mass


def serep_reduce(
    stiffness: np.ndarray,
    mass: np.ndarray,
    slave_dofs: np.ndarray,
    master_nodes_one_based: list[int] | tuple[int, ...],
    dofs_per_master_node: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the current SEREP reduction formula.

    Requires SciPy because the legacy implementation solves a generalized
    eigenvalue problem with `scipy.linalg.eigh`.
    """

    from scipy.linalg import eigh

    reordered_stiffness, reordered_mass = _reorder_master_slave_blocks(
        stiffness,
        mass,
        slave_dofs,
    )
    eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
    del eigenvalues

    for mode_index in range(eigenvectors.shape[1]):
        max_value = np.max(np.abs(eigenvectors[:, mode_index]))
        eigenvectors[:, mode_index] /= max_value

    master_size = dofs_per_master_node * len(master_nodes_one_based)
    modes = eigenvectors[:, 0:master_size]
    transformation = modes @ np.linalg.inv(modes[0:master_size, 0:master_size])
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation

    return reduced_mass, reduced_stiffness, transformation


def guyan_static_reduce(
    stiffness: np.ndarray,
    mass: np.ndarray,
    slave_dofs: np.ndarray,
    master_nodes_one_based: list[int] | tuple[int, ...],
    dofs_per_master_node: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run static condensation with the same `[master, slave]` ordering.

    This keeps the retained master DOFs exactly equal to the reduced coordinates
    and avoids inverting the SEREP master modal block. It is useful for high
    control-point counts where the square SEREP interpolation matrix becomes
    numerically singular.
    """

    reordered_stiffness, reordered_mass = _reorder_master_slave_blocks(
        stiffness,
        mass,
        slave_dofs,
    )
    master_size = dofs_per_master_node * len(master_nodes_one_based)
    stiffness_slave_slave = reordered_stiffness[master_size:, master_size:]
    stiffness_slave_master = reordered_stiffness[master_size:, :master_size]
    slave_transform = -np.linalg.solve(stiffness_slave_slave, stiffness_slave_master)
    transformation = np.vstack([np.eye(master_size), slave_transform])
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation

    return reduced_mass, reduced_stiffness, transformation


def guyan_static_reduce_ordered(
    stiffness: np.ndarray,
    mass: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run static condensation preserving the caller's master DOF order."""

    slave_dofs = np.sort(slave_dofs)
    reordered_stiffness, reordered_mass = _reorder_dof_blocks(
        stiffness,
        mass,
        np.asarray(master_dofs, dtype=int),
        slave_dofs,
    )
    master_size = len(master_dofs)
    stiffness_slave_slave = reordered_stiffness[master_size:, master_size:]
    stiffness_slave_master = reordered_stiffness[master_size:, :master_size]
    slave_transform = -np.linalg.solve(stiffness_slave_slave, stiffness_slave_master)
    transformation = np.vstack([np.eye(master_size), slave_transform])
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation

    return reduced_mass, reduced_stiffness, transformation


def serep_reduce_robust_ordered(
    stiffness: np.ndarray,
    mass: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    mode_multiplier: float = 3.0,
    rcond: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ordered SEREP with an overdetermined SVD pseudoinverse.

    Unlike the legacy square SEREP formula, this keeps the caller's master DOF
    order, uses more modes than master DOFs when available, and maps master
    coordinates to modal coordinates with an SVD pseudoinverse. This removes
    the direct inversion of the nearly singular square master modal block.
    """

    from scipy.linalg import eigh

    slave_dofs = np.sort(slave_dofs)
    reordered_stiffness, reordered_mass = _reorder_dof_blocks(
        stiffness,
        mass,
        np.asarray(master_dofs, dtype=int),
        slave_dofs,
    )
    _eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)

    master_size = len(master_dofs)
    requested_modes = int(np.ceil(master_size * mode_multiplier))
    mode_count = min(max(master_size, requested_modes), eigenvectors.shape[1])
    modes = eigenvectors[:, :mode_count]
    master_modes = modes[:master_size, :]
    transformation = modes @ np.linalg.pinv(master_modes, rcond=rcond)
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation

    return reduced_mass, reduced_stiffness, transformation


def serep_reduce_ridge_ordered(
    stiffness: np.ndarray,
    mass: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    relative_lambda: float = 1.0e-16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ordered square SEREP with Tikhonov-regularized inversion.

    This keeps the same number of modes as master DOFs, but replaces the direct
    inverse of the ill-conditioned master modal block with a tiny ridge solve:

    ``(Phi_m.T Phi_m + lambda I)^-1 Phi_m.T``.
    """

    from scipy.linalg import eigh

    slave_dofs = np.sort(slave_dofs)
    reordered_stiffness, reordered_mass = _reorder_dof_blocks(
        stiffness,
        mass,
        np.asarray(master_dofs, dtype=int),
        slave_dofs,
    )
    _eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
    for mode_index in range(eigenvectors.shape[1]):
        max_value = np.max(np.abs(eigenvectors[:, mode_index]))
        if max_value > 0.0:
            eigenvectors[:, mode_index] /= max_value

    master_size = len(master_dofs)
    modes = eigenvectors[:, :master_size]
    master_modes = modes[:master_size, :]
    normal_matrix = master_modes.T @ master_modes
    if relative_lambda == 0.0:
        mapping = np.linalg.inv(master_modes)
    else:
        scale = np.linalg.norm(normal_matrix, ord=2)
        ridge = relative_lambda * scale * np.eye(master_size)
        mapping = np.linalg.solve(normal_matrix + ridge, master_modes.T)
    transformation = modes @ mapping
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation

    return reduced_mass, reduced_stiffness, transformation


def serep_expansion_transform(
    stiffness: np.ndarray,
    mass: np.ndarray,
    slave_dofs: np.ndarray,
    master_nodes_one_based: list[int] | tuple[int, ...],
    dofs_per_master_node: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the current SEREP expansion transform formula."""

    from scipy.linalg import eigh

    reordered_stiffness, reordered_mass = _reorder_master_slave_blocks(
        stiffness,
        mass,
        slave_dofs,
    )
    eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
    del eigenvalues

    for mode_index in range(reordered_mass.shape[1]):
        norm = np.sqrt(
            np.dot(
                eigenvectors[:, mode_index].T,
                np.dot(reordered_mass, eigenvectors[:, mode_index]),
            )
        )
        eigenvectors[:, mode_index] /= norm

    master_size = dofs_per_master_node * len(master_nodes_one_based)
    transformation = eigenvectors[0:master_size, 0:master_size] @ np.linalg.pinv(
        eigenvectors[:, 0:master_size]
    )
    return reordered_mass, reordered_stiffness, transformation
