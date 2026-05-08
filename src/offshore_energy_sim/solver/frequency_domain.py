"""Frequency-domain linear solvers."""

from __future__ import annotations

import numpy as np


def dynamic_stiffness_matrix(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    omega: float | np.ndarray,
) -> np.ndarray:
    """Build `H = -omega**2 M - i omega C + K` for the MCK equation."""

    return -omega**2 * mass - 1j * omega * damping + stiffness


def solve_frequency_domain(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    force: np.ndarray,
    omega: float | np.ndarray,
) -> np.ndarray:
    """Solve the current frequency-domain equation for displacement.

    This preserves `DM_Assemble.solve_frequency_domain`: force is transposed
    before calling `numpy.linalg.solve`.
    """

    system = dynamic_stiffness_matrix(mass, damping, stiffness, omega)
    return np.linalg.solve(system, force.T)
