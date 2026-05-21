"""Structural retained-DOF and SEREP preparation for RODM cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.reduction import (
    guyan_static_reduce,
    guyan_static_reduce_ordered,
    reduce_matrix_dofs,
    separate_master_slave_dofs,
    serep_reduce,
    serep_reduce_ridge_ordered,
    serep_reduce_robust_ordered,
    transform_mass_matrix,
)
from offshore_energy_sim.structure.matrix_io import read_abaqus_matrix_dense


@dataclass(frozen=True)
class StructuralReductionResult:
    """Structural SEREP terms for the retained 5-DOF model."""

    master_nodes: list[int]
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray
    reverse_master_order_for_reconstruction: bool = True


def prepare_structural_reduction(
    case: RodmFrequencyCase,
    master_nodes: list[int],
) -> StructuralReductionResult:
    """Prepare structural retained-DOF matrices and SEREP transform.

    Numerical-result expectation: unchanged relative to the previous inline
    implementation in ``solve_rodm_frequency_case``.
    """

    mass_full = read_abaqus_matrix_dense(
        case.structural_matrices.mass,
        dofs_per_node=case.full_dofs_per_node,
    )
    stiffness_full = read_abaqus_matrix_dense(
        case.structural_matrices.stiffness,
        dofs_per_node=case.full_dofs_per_node,
    )

    # Full structural matrices: (total_nodes*6, total_nodes*6).
    # Retained structural matrices: (total_nodes*5, total_nodes*5).
    mass_retained = reduce_matrix_dofs(
        mass_full,
        case.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    stiffness_retained = reduce_matrix_dofs(
        stiffness_full,
        case.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    mass_retained = transform_mass_matrix(mass_retained, beta=case.mass_blend_beta)

    master_dofs, slave_dofs = separate_master_slave_dofs(
        case.total_nodes,
        master_nodes,
        dofs_per_node=case.retained_dofs_per_node,
    )
    if case.structural_reduction_method == "serep":
        reduced_mass, reduced_stiffness, transformation = serep_reduce(
            stiffness_retained,
            mass_retained,
            slave_dofs,
            master_nodes,
            dofs_per_master_node=case.retained_dofs_per_node,
        )
    elif case.structural_reduction_method == "guyan_static":
        if case.preserve_master_order:
            reduced_mass, reduced_stiffness, transformation = guyan_static_reduce_ordered(
                stiffness_retained,
                mass_retained,
                master_dofs,
                slave_dofs,
            )
        else:
            reduced_mass, reduced_stiffness, transformation = guyan_static_reduce(
                stiffness_retained,
                mass_retained,
                slave_dofs,
                master_nodes,
                dofs_per_master_node=case.retained_dofs_per_node,
            )
    elif case.structural_reduction_method == "serep_robust":
        reduced_mass, reduced_stiffness, transformation = serep_reduce_robust_ordered(
            stiffness_retained,
            mass_retained,
            master_dofs,
            slave_dofs,
            mode_multiplier=case.robust_serep_mode_multiplier,
            rcond=case.robust_serep_rcond,
        )
    elif case.structural_reduction_method == "serep_ridge":
        reduced_mass, reduced_stiffness, transformation = serep_reduce_ridge_ordered(
            stiffness_retained,
            mass_retained,
            master_dofs,
            slave_dofs,
            relative_lambda=case.serep_ridge_relative_lambda,
        )
    else:
        raise ValueError(
            "Unsupported structural_reduction_method: "
            f"{case.structural_reduction_method!r}"
        )

    return StructuralReductionResult(
        master_nodes=master_nodes,
        master_dofs=master_dofs,
        slave_dofs=slave_dofs,
        transformation=transformation,
        reduced_mass=reduced_mass,
        reduced_stiffness=reduced_stiffness,
        reverse_master_order_for_reconstruction=not (
            case.preserve_master_order
            or case.structural_reduction_method in {"serep_robust", "serep_ridge"}
        ),
    )
