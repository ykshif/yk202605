"""External WEC-Sim-like time-domain adapter layer for RODM outputs.

This package is intentionally outside the RODM frequency-domain solver path.
It reads exported hydrodynamic/frequency-domain data, prepares optional
Cummins-style time-domain inputs, and writes adapter-owned diagnostics.
"""

from offshore_energy_sim.time_domain_adapter.hydrodynamic_extrapolation import (
    ExtrapolatedHydrodynamicData,
    HydrodynamicExtrapolationConfig,
    build_extended_omega_grid,
    frequency_grid_diagnostics,
    extrapolate_frequency_series,
    hydrodynamic_array_diagnostics,
    max_abs_difference_inside_original_range,
    extrapolate_hydrodynamic_data,
)
from offshore_energy_sim.time_domain_adapter.mooring import (
    build_corner_mooring_reduced_stiffness,
    corner_mooring_diagonal_stiffness,
    corner_node_ids_for_regular_grid,
    project_diagonal_stiffness_to_reduced,
)
from offshore_energy_sim.time_domain_adapter.radiation_kernel import (
    RadiationKernelDiagnostics,
    build_radiation_kernel,
    compare_radiation_kernels,
    radiation_kernel_diagnostics,
)
from offshore_energy_sim.time_domain_adapter.state_space_radiation import (
    DiscreteStateSpaceRadiationModel,
    StateSpaceRadiationModel,
    default_real_poles,
    evaluate_era_markov_parameters,
    evaluate_era_radiation_kernel,
    evaluate_state_space_radiation_kernel,
    fit_era_state_space_radiation,
    fit_common_pole_state_space_radiation,
    load_discrete_state_space_radiation_model,
    save_discrete_state_space_radiation_model,
    simulate_era_memory_force,
    simulate_state_space_memory_force,
    state_space_radiation_coefficients,
)
from offshore_energy_sim.time_domain_adapter.state_space_solver import (
    StateSpaceRadiationLinearSystem,
    solve_state_space_radiation_linear_system,
    solve_state_space_radiation_linear_system_rk4,
)
from offshore_energy_sim.time_domain_adapter.wecsim_like_solver import (
    MooringLinearization,
    ResolvedMooringLinearization,
    WecSimLikeRadiationConfig,
    WecSimLikeTimeDomainResult,
    solve_rodm_wecsim_like_time_domain,
)

__all__ = [
    "ExtrapolatedHydrodynamicData",
    "HydrodynamicExtrapolationConfig",
    "DiscreteStateSpaceRadiationModel",
    "MooringLinearization",
    "ResolvedMooringLinearization",
    "RadiationKernelDiagnostics",
    "StateSpaceRadiationModel",
    "StateSpaceRadiationLinearSystem",
    "WecSimLikeRadiationConfig",
    "WecSimLikeTimeDomainResult",
    "build_extended_omega_grid",
    "build_corner_mooring_reduced_stiffness",
    "build_radiation_kernel",
    "compare_radiation_kernels",
    "corner_mooring_diagonal_stiffness",
    "corner_node_ids_for_regular_grid",
    "default_real_poles",
    "evaluate_era_markov_parameters",
    "evaluate_era_radiation_kernel",
    "evaluate_state_space_radiation_kernel",
    "extrapolate_hydrodynamic_data",
    "extrapolate_frequency_series",
    "fit_era_state_space_radiation",
    "fit_common_pole_state_space_radiation",
    "frequency_grid_diagnostics",
    "hydrodynamic_array_diagnostics",
    "load_discrete_state_space_radiation_model",
    "max_abs_difference_inside_original_range",
    "project_diagonal_stiffness_to_reduced",
    "radiation_kernel_diagnostics",
    "save_discrete_state_space_radiation_model",
    "simulate_era_memory_force",
    "simulate_state_space_memory_force",
    "solve_rodm_wecsim_like_time_domain",
    "solve_state_space_radiation_linear_system",
    "solve_state_space_radiation_linear_system_rk4",
    "state_space_radiation_coefficients",
]
