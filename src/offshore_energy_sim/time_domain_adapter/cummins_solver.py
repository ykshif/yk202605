"""Adapter-owned Cummins equation wrappers.

This module keeps WEC-Sim/Cummins time-domain solving outside the RODM
frequency-domain main program. It consumes already exported matrices and
frequency-domain hydrodynamic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.time_domain.solver import LinearTimeDomainResult, solve_linear_time_domain


@dataclass(frozen=True)
class CumminsLinearSystem:
    """Matrices for a linear Cummins time-domain solve."""

    structural_mass: np.ndarray
    structural_stiffness: np.ndarray
    added_mass_infinite: np.ndarray
    radiation_kernel: np.ndarray
    time: np.ndarray
    force: np.ndarray
    linear_damping: np.ndarray | None = None
    hydrostatic_stiffness: np.ndarray | None = None
    mooring_stiffness: np.ndarray | None = None
    mooring_damping: np.ndarray | None = None
    mooring_pretension: np.ndarray | None = None


def solve_cummins_linear_system(
    system: CumminsLinearSystem,
    *,
    radiation_convolution_rule: str = "trapezoidal",
) -> LinearTimeDomainResult:
    """Solve ``(M + A_inf) qdd + K*q + int K_r*qdot = F``.

    The function is intentionally a thin adapter over the existing generic
    Newmark/Cummins integrator. It does not call or modify RODM internals.
    """

    mass = np.asarray(system.structural_mass, dtype=float) + np.asarray(
        system.added_mass_infinite,
        dtype=float,
    )
    stiffness = np.asarray(system.structural_stiffness, dtype=float).copy()
    if system.hydrostatic_stiffness is not None:
        stiffness = stiffness + np.asarray(system.hydrostatic_stiffness, dtype=float)
    if system.mooring_stiffness is not None:
        stiffness = stiffness + np.asarray(system.mooring_stiffness, dtype=float)
    damping = (
        np.zeros_like(mass)
        if system.linear_damping is None
        else np.asarray(system.linear_damping, dtype=float)
    )
    if system.mooring_damping is not None:
        damping = damping + np.asarray(system.mooring_damping, dtype=float)
    force = np.asarray(system.force, dtype=float)
    if system.mooring_pretension is not None:
        force = force + np.asarray(system.mooring_pretension, dtype=float).reshape(1, -1)
    return solve_linear_time_domain(
        mass,
        damping,
        stiffness,
        force,
        system.time,
        radiation_irf=system.radiation_kernel,
        radiation_convolution_rule=radiation_convolution_rule,
    )
