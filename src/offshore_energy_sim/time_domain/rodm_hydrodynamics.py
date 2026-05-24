"""RODM hydrodynamic preprocessing for time-domain simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.hydrodynamics import (
    prepare_hydrodynamic_terms,
    reverse_hydrodynamic_node_order_force,
    reverse_hydrodynamic_node_order_matrix,
)
from offshore_energy_sim.reduction import reduce_force_dofs, reduce_matrix_dofs
from offshore_energy_sim.time_domain.cases import TimeDomainSimulationConfig
from offshore_energy_sim.time_domain.hydrodynamic_memory import (
    apply_radiation_frequency_window,
    estimate_infinite_frequency_added_mass,
    estimate_infinite_frequency_added_mass_from_irf,
    project_matrix_series_positive_semidefinite,
    project_symmetric_positive_semidefinite,
    radiation_coefficients_from_discrete_irf,
    radiation_coefficients_from_irf,
    radiation_irf_from_damping,
)


@dataclass(frozen=True)
class TimeDomainHydrodynamicTerms:
    """Hydrodynamic terms aligned to the RODM retained master DOFs."""

    omega: float
    omega_grid: np.ndarray
    added_mass: np.ndarray
    radiation_damping: np.ndarray
    hydrostatic_stiffness: np.ndarray
    wave_force: np.ndarray
    wave_force_series: np.ndarray | None = None
    added_mass_infinite: np.ndarray | None = None
    radiation_irf: np.ndarray | None = None
    radiation_irf_time: np.ndarray | None = None
    residual_added_mass: np.ndarray | None = None
    residual_radiation_damping: np.ndarray | None = None


def _reduced_matrix_series(values: np.ndarray, case: RodmFrequencyCase) -> np.ndarray:
    """Reduce a frequency-major hydrodynamic matrix series to retained DOFs."""

    matrices = np.asarray(values, dtype=float)
    if matrices.ndim != 3:
        raise ValueError("hydrodynamic matrix series must have shape (n_omega, ndof, ndof)")
    reduced = [
        reduce_matrix_dofs(
            matrices[index],
            case.hydrodynamic_nodes,
            [case.hydrodynamic_dof_to_remove_zero_based],
        )
        for index in range(matrices.shape[0])
    ]
    series = np.stack(reduced, axis=0)
    if case.reverse_hydrodynamic_node_order:
        series = np.stack(
            [
                reverse_hydrodynamic_node_order_matrix(
                    matrix,
                    case.hydrodynamic_nodes,
                    case.retained_dofs_per_node,
                )
                for matrix in series
            ],
            axis=0,
        )
    return series


def _selected_wave_force(case: RodmFrequencyCase, dataset) -> np.ndarray:
    """Return the selected reduced complex wave-force vector."""

    froude_krylov = _force_values_at_frequency(dataset, "Froude_Krylov_force", case.frequency_index)
    diffraction = _force_values_at_frequency(dataset, "diffraction_force", case.frequency_index)
    wave_force = reduce_force_dofs(
        froude_krylov + diffraction,
        case.hydrodynamic_nodes,
        case.hydrodynamic_dof_to_remove_zero_based,
    ).reshape(1, case.hydrodynamic_nodes * case.retained_dofs_per_node)
    if case.reverse_hydrodynamic_node_order:
        wave_force = reverse_hydrodynamic_node_order_force(
            wave_force,
            case.hydrodynamic_nodes,
            case.retained_dofs_per_node,
        )
    return wave_force


def _force_values_at_frequency(dataset, variable_name: str, frequency_index: int) -> np.ndarray:
    """Return one wave-direction force vector at a frequency index."""

    values = dataset[variable_name].isel(omega=frequency_index)
    if "wave_direction" in values.dims:
        values = values.isel(wave_direction=0)
    return values.values


def _wave_force_series(dataset, case: RodmFrequencyCase, order: np.ndarray) -> np.ndarray:
    """Return reduced complex wave-force transfer functions for all frequencies."""

    forces = []
    for frequency_index in order:
        force = reduce_force_dofs(
            _force_values_at_frequency(dataset, "Froude_Krylov_force", int(frequency_index))
            + _force_values_at_frequency(dataset, "diffraction_force", int(frequency_index)),
            case.hydrodynamic_nodes,
            case.hydrodynamic_dof_to_remove_zero_based,
        )
        force = force.reshape(1, case.hydrodynamic_nodes * case.retained_dofs_per_node)
        if case.reverse_hydrodynamic_node_order:
            force = reverse_hydrodynamic_node_order_force(
                force,
                case.hydrodynamic_nodes,
                case.retained_dofs_per_node,
            )
        forces.append(force.reshape(-1))
    return np.stack(forces, axis=0)


def _memory_time_values(
    config: TimeDomainSimulationConfig,
    selected_omega: float,
) -> np.ndarray:
    """Return the kernel time vector for direct-convolution radiation memory."""

    if config.memory_duration is None:
        if selected_omega <= 0.0:
            raise ValueError("default memory_duration requires a positive selected omega")
        period = 2.0 * np.pi / selected_omega
        memory_duration = min(config.duration, 5.0 * period)
    else:
        memory_duration = config.memory_duration
    step_count = int(np.floor(memory_duration / config.time_step))
    return np.arange(step_count + 1, dtype=float) * config.time_step


def prepare_rodm_time_domain_hydrodynamic_terms(
    case: RodmFrequencyCase,
    dataset,
    config: TimeDomainSimulationConfig,
) -> TimeDomainHydrodynamicTerms:
    """Prepare RODM hydrodynamic terms for the selected time-domain model.

    ``radiation_model='constant'`` returns the same selected-frequency terms as
    the frequency-domain bridge. ``radiation_model='direct_convolution'`` also
    builds Cummins radiation-memory terms from the full BEM frequency grid.
    """

    selected = prepare_hydrodynamic_terms(case, dataset)
    selected_omega = float(np.asarray(selected.omega, dtype=float).reshape(-1)[0])
    omega_grid = np.asarray(dataset.omega.values, dtype=float).reshape(-1)
    if config.radiation_model == "constant":
        return TimeDomainHydrodynamicTerms(
            omega=selected_omega,
            omega_grid=omega_grid,
            added_mass=selected.added_mass,
            radiation_damping=selected.radiation_damping,
            hydrostatic_stiffness=selected.hydrostatic_stiffness,
            wave_force=selected.wave_force,
            wave_force_series=None,
        )

    if omega_grid.size < 2:
        raise ValueError("direct_convolution radiation model requires at least two frequencies")
    order = np.argsort(omega_grid)
    omega_sorted = omega_grid[order]
    if np.any(np.diff(omega_sorted) <= 0.0):
        raise ValueError("hydrodynamic omega grid must contain unique increasing values")

    added_mass_series = _reduced_matrix_series(dataset["added_mass"].values, case)[order]
    damping_series = _reduced_matrix_series(dataset["radiation_damping"].values, case)[order]
    wave_force_series = _wave_force_series(dataset, case, order)
    if config.radiation_passivity_correction == "clip_negative_eigenvalues":
        damping_series = project_matrix_series_positive_semidefinite(damping_series)
    damping_for_irf = apply_radiation_frequency_window(
        omega_sorted,
        damping_series,
        window=config.radiation_frequency_window,
        start_omega=config.radiation_window_start_omega,
        stop_omega=config.radiation_window_stop_omega,
    )
    irf_time = _memory_time_values(config, selected_omega)
    radiation_irf = radiation_irf_from_damping(
        omega_sorted,
        damping_for_irf,
        irf_time,
        damping_convention=config.damping_convention,
    )
    if config.infinite_added_mass_method == "ogilvie":
        added_mass_infinite = estimate_infinite_frequency_added_mass_from_irf(
            omega_sorted,
            added_mass_series,
            radiation_irf,
            irf_time,
            tail_count=config.added_mass_tail_count,
        )
    else:
        added_mass_infinite = estimate_infinite_frequency_added_mass(
            omega_sorted,
            added_mass_series,
            method=config.infinite_added_mass_method,
            tail_count=config.added_mass_tail_count,
        )
    if config.radiation_passivity_correction == "clip_negative_eigenvalues":
        added_mass_infinite = project_symmetric_positive_semidefinite(added_mass_infinite)
    residual_added_mass = None
    residual_radiation_damping = None
    if config.radiation_residual_model == "selected_frequency":
        reconstructed_added_mass, reconstructed_damping = radiation_coefficients_from_discrete_irf(
            selected_omega,
            radiation_irf,
            irf_time,
            added_mass_infinite=added_mass_infinite,
            convolution_rule=config.radiation_convolution_rule,
        )
        residual_added_mass = selected.added_mass - reconstructed_added_mass
        residual_radiation_damping = selected.radiation_damping - reconstructed_damping

    return TimeDomainHydrodynamicTerms(
        omega=selected_omega,
        omega_grid=omega_sorted,
        added_mass=selected.added_mass,
        radiation_damping=selected.radiation_damping,
        hydrostatic_stiffness=selected.hydrostatic_stiffness,
        wave_force=_selected_wave_force(case, dataset),
        wave_force_series=wave_force_series,
        added_mass_infinite=added_mass_infinite,
        radiation_irf=radiation_irf,
        radiation_irf_time=irf_time,
        residual_added_mass=residual_added_mass,
        residual_radiation_damping=residual_radiation_damping,
    )
