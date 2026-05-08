"""Frequency-domain hydrodynamic terms for RODM cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.reduction import reduce_force_dofs, reduce_matrix_dofs


@dataclass(frozen=True)
class HydrodynamicTerms:
    """Reduced hydrodynamic terms aligned to the retained master DOFs."""

    added_mass: np.ndarray
    radiation_damping: np.ndarray
    hydrostatic_stiffness: np.ndarray
    wave_force: np.ndarray
    omega: float | np.ndarray


def reverse_hydrodynamic_node_order_matrix(
    matrix: np.ndarray,
    node_count: int,
    dofs_per_node: int,
) -> np.ndarray:
    """Reverse hydrodynamic node blocks while preserving local DOF order."""

    order = np.arange(node_count * dofs_per_node).reshape(node_count, dofs_per_node)[::-1].ravel()
    return matrix[np.ix_(order, order)]


def reverse_hydrodynamic_node_order_force(
    force: np.ndarray,
    node_count: int,
    dofs_per_node: int,
) -> np.ndarray:
    """Reverse hydrodynamic force node blocks while preserving local DOF order."""

    return force.reshape(node_count, dofs_per_node)[::-1].reshape(
        1,
        node_count * dofs_per_node,
    )


def prepare_hydrodynamic_terms(case: RodmFrequencyCase, dataset) -> HydrodynamicTerms:
    """Prepare reduced hydrodynamic matrices and wave force.

    Numerical-result expectation: unchanged for default cases. Hydrodynamic
    node reversal remains controlled only by
    ``case.reverse_hydrodynamic_node_order``.
    """

    omega_values = dataset.omega.values
    omega = omega_values[case.frequency_index] if np.ndim(omega_values) else omega_values

    added_mass = reduce_matrix_dofs(
        dataset["added_mass"][case.frequency_index].values,
        case.hydrodynamic_nodes,
        [case.hydrodynamic_dof_to_remove_zero_based],
    )
    radiation_damping = reduce_matrix_dofs(
        dataset["radiation_damping"][case.frequency_index].values,
        case.hydrodynamic_nodes,
        [case.hydrodynamic_dof_to_remove_zero_based],
    )
    hydrostatic_stiffness = reduce_matrix_dofs(
        dataset["hydrostatic_stiffness"].values,
        case.hydrodynamic_nodes,
        [case.hydrodynamic_dof_to_remove_zero_based],
    )

    froude_krylov = dataset["Froude_Krylov_force"][case.frequency_index].values
    diffraction = dataset["diffraction_force"][case.frequency_index].values
    wave_force = reduce_force_dofs(
        froude_krylov + diffraction,
        case.hydrodynamic_nodes,
        case.hydrodynamic_dof_to_remove_zero_based,
    ).reshape(1, case.hydrodynamic_nodes * case.retained_dofs_per_node)

    if case.reverse_hydrodynamic_node_order:
        added_mass = reverse_hydrodynamic_node_order_matrix(
            added_mass,
            case.hydrodynamic_nodes,
            case.retained_dofs_per_node,
        )
        radiation_damping = reverse_hydrodynamic_node_order_matrix(
            radiation_damping,
            case.hydrodynamic_nodes,
            case.retained_dofs_per_node,
        )
        hydrostatic_stiffness = reverse_hydrodynamic_node_order_matrix(
            hydrostatic_stiffness,
            case.hydrodynamic_nodes,
            case.retained_dofs_per_node,
        )
        wave_force = reverse_hydrodynamic_node_order_force(
            wave_force,
            case.hydrodynamic_nodes,
            case.retained_dofs_per_node,
        )

    return HydrodynamicTerms(
        added_mass=added_mass,
        radiation_damping=radiation_damping,
        hydrostatic_stiffness=hydrostatic_stiffness,
        wave_force=wave_force,
        omega=omega,
    )
