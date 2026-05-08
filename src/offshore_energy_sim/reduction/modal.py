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
