"""Case-level RODM frequency-domain orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.core.dependencies import require_optional_dependencies
from offshore_energy_sim.hydrodynamics import (
    open_hydrodynamic_dataset,
    prepare_hydrodynamic_terms,
)
from offshore_energy_sim.response import reconstruct_global_response
from offshore_energy_sim.solver.frequency_domain import solve_frequency_domain
from offshore_energy_sim.structure import (
    calculate_node_positions,
    prepare_structural_reduction,
)


@dataclass(frozen=True)
class RodmFrequencyResult:
    """Outputs from the current RODM frequency-domain workflow."""

    global_displacement: np.ndarray
    master_displacement: np.ndarray
    master_nodes: list[int]
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray
    added_mass: np.ndarray
    radiation_damping: np.ndarray
    wave_force: np.ndarray
    omega: float | np.ndarray


def solve_rodm_frequency_case(case: RodmFrequencyCase) -> RodmFrequencyResult:
    """Run the current RODM frequency-domain workflow for one case.

    The implementation follows `DM_Method.perform_RODM_reduce_order_model`:
    structural matrices are read, full 6-DOF nodes are reduced to retained DOFs,
    SEREP builds the master transformation, hydrodynamic matrices are reduced,
    and the MCK frequency-domain equation is solved.

    Numerical-result expectation: unchanged relative to the legacy workflow
    when the same files and dependencies are used.
    """

    require_optional_dependencies(("xarray", "capytaine", "scipy"))
    if not case.use_hydrostatic:
        raise NotImplementedError("FEM spring hydrostatic alternative is not migrated yet.")

    if case.master_nodes_one_based is None:
        master_nodes = calculate_node_positions(
            case.master_node_rule.first_node,
            case.master_node_rule.node_interval,
            case.master_node_rule.count,
        )
    else:
        master_nodes = list(case.master_nodes_one_based)

    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset, merge_complex=True)
    try:
        structural = prepare_structural_reduction(case, master_nodes)
        hydrodynamic = prepare_hydrodynamic_terms(case, dataset)

        effective_mass = hydrodynamic.added_mass + structural.reduced_mass
        effective_damping = hydrodynamic.radiation_damping
        effective_stiffness = hydrodynamic.hydrostatic_stiffness + structural.reduced_stiffness

        master_displacement = solve_frequency_domain(
            effective_mass,
            effective_damping,
            effective_stiffness,
            hydrodynamic.wave_force,
            hydrodynamic.omega,
        )
        global_displacement = reconstruct_global_response(
            structural.transformation,
            master_displacement,
            structural.master_dofs,
            structural.slave_dofs,
            reverse_master_order=structural.reverse_master_order_for_reconstruction,
        )

        return RodmFrequencyResult(
            global_displacement=global_displacement,
            master_displacement=master_displacement,
            master_nodes=structural.master_nodes,
            master_dofs=structural.master_dofs,
            slave_dofs=structural.slave_dofs,
            transformation=structural.transformation,
            reduced_mass=structural.reduced_mass,
            reduced_stiffness=structural.reduced_stiffness,
            added_mass=hydrodynamic.added_mass,
            radiation_damping=hydrodynamic.radiation_damping,
            wave_force=hydrodynamic.wave_force,
            omega=hydrodynamic.omega,
        )
    finally:
        dataset.close()
