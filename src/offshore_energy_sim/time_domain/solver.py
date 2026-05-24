"""Linear time-domain solvers for RODM hydroelastic systems."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.core.dependencies import require_optional_dependencies
from offshore_energy_sim.hydrodynamics import (
    open_hydrodynamic_dataset,
)
from offshore_energy_sim.mooring import ReducedMooringTerms
from offshore_energy_sim.response import reconstruct_global_response
from offshore_energy_sim.structure import (
    StructuralReductionResult,
    calculate_node_positions,
    prepare_structural_reduction,
)
from offshore_energy_sim.time_domain.cases import TimeDomainSimulationConfig
from offshore_energy_sim.time_domain.excitation import (
    external_force_time_series,
    harmonic_force_time_series,
    random_wave_phases,
    spectral_wave_amplitudes,
    spectral_wave_force_time_series,
    wave_elevation_time_series,
    wave_spectrum_density,
)
from offshore_energy_sim.time_domain.rodm_hydrodynamics import (
    prepare_rodm_time_domain_hydrodynamic_terms,
)


@dataclass(frozen=True)
class LinearTimeDomainResult:
    """Time histories from a fixed-step linear MCK integration."""

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    memory_force: np.ndarray


@dataclass(frozen=True)
class RodmTimeDomainResult:
    """Time-domain result for one RODM frequency-case input."""

    time: np.ndarray
    master_displacement: np.ndarray
    master_velocity: np.ndarray
    master_acceleration: np.ndarray
    global_displacement: np.ndarray
    memory_force: np.ndarray
    master_nodes: list[int]
    omega: float | np.ndarray
    radiation_model: str = "constant"
    radiation_irf_time: np.ndarray | None = None
    radiation_irf: np.ndarray | None = None
    added_mass_infinite: np.ndarray | None = None
    residual_added_mass: np.ndarray | None = None
    residual_radiation_damping: np.ndarray | None = None
    excitation_model: str = "regular_wave"
    excitation_force: np.ndarray | None = None
    wave_elevation: np.ndarray | None = None
    wave_component_omega: np.ndarray | None = None
    wave_component_amplitude: np.ndarray | None = None
    wave_component_phase: np.ndarray | None = None
    mooring_reduced_stiffness: np.ndarray | None = None
    mooring_reduced_damping: np.ndarray | None = None
    mooring_reduced_pretension: np.ndarray | None = None
    mooring_metadata: Mapping[str, object] | None = None


def _as_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Validate one dense square matrix."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def _as_time_matrix(values: np.ndarray, ndof: int, name: str) -> np.ndarray:
    """Validate a time-major real matrix."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != ndof:
        raise ValueError(f"{name} must have shape (n_time, {ndof})")
    return array


def direct_convolution_memory_force(
    velocity_history: np.ndarray,
    radiation_irf: np.ndarray,
    time_step: float,
    step_index: int,
    *,
    convolution_rule: str = "rectangular",
) -> np.ndarray:
    """Return explicit radiation-memory force at one step.

    The zero-lag term is intentionally omitted so the memory force depends only
    on already known velocities. This keeps the Newmark step linear and is a
    conservative first implementation; a later implicit variant can fold the
    zero-lag term into the damping matrix.
    """

    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    velocity = np.asarray(velocity_history, dtype=float)
    kernel = np.asarray(radiation_irf, dtype=float)
    if velocity.ndim != 2:
        raise ValueError("velocity_history must have shape (n_time, ndof)")
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_irf must have shape (n_kernel, ndof, ndof)")
    if kernel.shape[1] != velocity.shape[1]:
        raise ValueError("radiation_irf ndof must match velocity_history")
    if step_index < 0 or step_index >= velocity.shape[0]:
        raise ValueError("step_index is outside velocity_history")
    rule = str(convolution_rule).lower()
    if rule not in {"rectangular", "trapezoidal"}:
        raise ValueError("convolution_rule must be 'rectangular' or 'trapezoidal'")

    max_lag = min(step_index, kernel.shape[0] - 1)
    kernel_slice = kernel[1 : max_lag + 1]
    velocity_slice = velocity[step_index - max_lag : step_index][::-1]
    if rule == "trapezoidal" and max_lag > 0:
        weights = np.ones(max_lag, dtype=float)
        weights[-1] = 0.5
        return time_step * np.einsum("k,kij,kj->i", weights, kernel_slice, velocity_slice)
    return time_step * np.einsum("kij,kj->i", kernel_slice, velocity_slice)


def _linear_interpolate_rows(values: np.ndarray, index: int, theta: float) -> np.ndarray:
    """Return row-linear interpolation between ``index`` and ``index + 1``."""

    return (1.0 - theta) * values[index] + theta * values[index + 1]


def _radiation_zero_lag_damping(
    damping: np.ndarray,
    radiation_irf: np.ndarray | None,
    time_step: float,
    *,
    implicit_radiation_zero_lag: bool,
    radiation_convolution_rule: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate optional IRF input and return the implicit zero-lag damping."""

    radiation_zero_lag = np.zeros_like(damping)
    if radiation_irf is None:
        return radiation_zero_lag, None
    rule = str(radiation_convolution_rule).lower()
    if rule not in {"rectangular", "trapezoidal"}:
        raise ValueError("radiation_convolution_rule must be 'rectangular' or 'trapezoidal'")
    kernel = np.asarray(radiation_irf, dtype=float)
    if kernel.ndim != 3 or kernel.shape[1:] != damping.shape:
        raise ValueError("radiation_irf must have shape (n_kernel, ndof, ndof)")
    if implicit_radiation_zero_lag and kernel.shape[0] > 0:
        zero_lag_weight = 0.5 if rule == "trapezoidal" else 1.0
        radiation_zero_lag = zero_lag_weight * time_step * kernel[0]
    return radiation_zero_lag, kernel


def solve_linear_time_domain_rk4(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    force: np.ndarray,
    time: np.ndarray,
    *,
    initial_displacement: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    radiation_irf: np.ndarray | None = None,
    implicit_radiation_zero_lag: bool = True,
    radiation_convolution_rule: str = "rectangular",
) -> LinearTimeDomainResult:
    """Solve the reduced linear system with an explicit fourth-order RK method.

    Radiation history is treated consistently with the existing Newmark path:
    only already known grid-point velocity history enters the convolution. The
    resulting history force is linearly interpolated inside one RK step.
    """

    M = _as_square_matrix(mass, "mass")
    C = _as_square_matrix(damping, "damping")
    K = _as_square_matrix(stiffness, "stiffness")
    if M.shape != C.shape or M.shape != K.shape:
        raise ValueError("mass, damping, and stiffness must have the same shape")
    ndof = M.shape[0]
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError("time must be strictly increasing")
    F = _as_time_matrix(force, ndof, "force")
    if F.shape[0] != time.size:
        raise ValueError("force first axis must match time")

    radiation_zero_lag, kernel = _radiation_zero_lag_damping(
        C,
        radiation_irf,
        dt,
        implicit_radiation_zero_lag=implicit_radiation_zero_lag,
        radiation_convolution_rule=radiation_convolution_rule,
    )
    C_step = C + radiation_zero_lag

    q = np.zeros((time.size, ndof), dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    memory = np.zeros_like(q)
    if initial_displacement is not None:
        q[0] = np.asarray(initial_displacement, dtype=float).reshape(ndof)
    if initial_velocity is not None:
        v[0] = np.asarray(initial_velocity, dtype=float).reshape(ndof)

    def acceleration(
        q_value: np.ndarray,
        v_value: np.ndarray,
        force_value: np.ndarray,
        history_memory: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.solve(M, force_value - history_memory - C_step @ v_value - K @ q_value)

    a[0] = acceleration(q[0], v[0], F[0], np.zeros(ndof, dtype=float))

    for index in range(time.size - 1):
        next_index = index + 1
        history_start = memory[index] - radiation_zero_lag @ v[index]
        history_end = np.zeros(ndof, dtype=float)
        if kernel is not None:
            history_end = direct_convolution_memory_force(
                v,
                kernel,
                dt,
                next_index,
                convolution_rule=radiation_convolution_rule,
            )

        def derivative(theta: float, q_value: np.ndarray, v_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            force_value = _linear_interpolate_rows(F, index, theta)
            history_value = (1.0 - theta) * history_start + theta * history_end
            return v_value, acceleration(q_value, v_value, force_value, history_value)

        k1_q, k1_v = derivative(0.0, q[index], v[index])
        k2_q, k2_v = derivative(0.5, q[index] + 0.5 * dt * k1_q, v[index] + 0.5 * dt * k1_v)
        k3_q, k3_v = derivative(0.5, q[index] + 0.5 * dt * k2_q, v[index] + 0.5 * dt * k2_v)
        k4_q, k4_v = derivative(1.0, q[index] + dt * k3_q, v[index] + dt * k3_v)

        q[next_index] = q[index] + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        v[next_index] = v[index] + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        memory[next_index] = history_end + radiation_zero_lag @ v[next_index]
        a[next_index] = acceleration(q[next_index], v[next_index], F[next_index], history_end)

    return LinearTimeDomainResult(
        time=time,
        displacement=q,
        velocity=v,
        acceleration=a,
        memory_force=memory,
    )


def solve_linear_time_domain(
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    force: np.ndarray,
    time: np.ndarray,
    *,
    initial_displacement: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    radiation_irf: np.ndarray | None = None,
    implicit_radiation_zero_lag: bool = True,
    radiation_convolution_rule: str = "rectangular",
    integrator: str = "newmark",
    newmark_beta: float = 0.25,
    newmark_gamma: float = 0.5,
) -> LinearTimeDomainResult:
    """Solve ``M qdd + C qd + K q + F_memory = F(t)``.

    The integrator is fixed-step Newmark average acceleration by default.
    Pass ``integrator="rk4"`` to use the explicit fourth-order Runge-Kutta
    reduced-space integrator.
    """

    method = str(integrator).lower()
    if method == "rk4":
        return solve_linear_time_domain_rk4(
            mass,
            damping,
            stiffness,
            force,
            time,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
            radiation_irf=radiation_irf,
            implicit_radiation_zero_lag=implicit_radiation_zero_lag,
            radiation_convolution_rule=radiation_convolution_rule,
        )
    if method != "newmark":
        raise ValueError("integrator must be 'newmark' or 'rk4'")

    M = _as_square_matrix(mass, "mass")
    C = _as_square_matrix(damping, "damping")
    K = _as_square_matrix(stiffness, "stiffness")
    if M.shape != C.shape or M.shape != K.shape:
        raise ValueError("mass, damping, and stiffness must have the same shape")
    ndof = M.shape[0]
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError("time must be strictly increasing")
    F = _as_time_matrix(force, ndof, "force")
    if F.shape[0] != time.size:
        raise ValueError("force first axis must match time")
    if newmark_beta <= 0.0 or newmark_gamma <= 0.0:
        raise ValueError("Newmark beta and gamma must be positive")

    q = np.zeros((time.size, ndof), dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    memory = np.zeros_like(q)
    if initial_displacement is not None:
        q[0] = np.asarray(initial_displacement, dtype=float).reshape(ndof)
    if initial_velocity is not None:
        v[0] = np.asarray(initial_velocity, dtype=float).reshape(ndof)

    radiation_zero_lag_damping, kernel = _radiation_zero_lag_damping(
        C,
        radiation_irf,
        dt,
        implicit_radiation_zero_lag=implicit_radiation_zero_lag,
        radiation_convolution_rule=radiation_convolution_rule,
    )
    C_step = C + radiation_zero_lag_damping

    a[0] = np.linalg.solve(M, F[0] - C_step @ v[0] - K @ q[0])
    effective = M + newmark_gamma * dt * C_step + newmark_beta * dt**2 * K

    for index in range(time.size - 1):
        next_index = index + 1
        history_memory = np.zeros(ndof, dtype=float)
        if kernel is not None:
            history_memory = direct_convolution_memory_force(
                v,
                kernel,
                dt,
                next_index,
                convolution_rule=radiation_convolution_rule,
            )

        q_predict = q[index] + dt * v[index] + dt**2 * (0.5 - newmark_beta) * a[index]
        v_predict = v[index] + dt * (1.0 - newmark_gamma) * a[index]
        rhs = F[next_index] - history_memory - C_step @ v_predict - K @ q_predict
        a[next_index] = np.linalg.solve(effective, rhs)
        q[next_index] = q_predict + newmark_beta * dt**2 * a[next_index]
        v[next_index] = v_predict + newmark_gamma * dt * a[next_index]
        memory[next_index] = history_memory + radiation_zero_lag_damping @ v[next_index]

    return LinearTimeDomainResult(
        time=time,
        displacement=q,
        velocity=v,
        acceleration=a,
        memory_force=memory,
    )


def solve_rodm_time_domain_case(
    case: RodmFrequencyCase,
    config: TimeDomainSimulationConfig,
    *,
    mooring_provider: Callable[
        [RodmFrequencyCase, StructuralReductionResult],
        ReducedMooringTerms | None,
    ]
    | None = None,
) -> RodmTimeDomainResult:
    """Run a single-frequency linear time-domain simulation for a RODM case.

    This first RODM time-domain path intentionally uses the same hydrodynamic
    frequency index as the frequency-domain solver. It is therefore a
    validation bridge: the fitted steady-state response should match
    ``solve_rodm_frequency_case`` before full radiation memory is enabled.
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
        hydrodynamic = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)
        omega = float(np.asarray(hydrodynamic.omega).reshape(-1)[0])
        time = config.time_values()

        if config.radiation_model == "constant":
            hydrodynamic_mass = hydrodynamic.added_mass
            effective_damping = hydrodynamic.radiation_damping
            radiation_irf = None
        else:
            if hydrodynamic.added_mass_infinite is None or hydrodynamic.radiation_irf is None:
                raise RuntimeError("direct_convolution hydrodynamic preprocessing is incomplete")
            mass_residual = (
                0.0
                if hydrodynamic.residual_added_mass is None
                else hydrodynamic.residual_added_mass
            )
            damping_residual = (
                np.zeros_like(hydrodynamic.radiation_damping)
                if hydrodynamic.residual_radiation_damping is None
                else hydrodynamic.residual_radiation_damping
            )
            hydrodynamic_mass = hydrodynamic.added_mass_infinite + mass_residual
            effective_damping = damping_residual
            radiation_irf = hydrodynamic.radiation_irf

        effective_mass = hydrodynamic_mass + structural.reduced_mass
        effective_stiffness = hydrodynamic.hydrostatic_stiffness + structural.reduced_stiffness
        mooring_terms = mooring_provider(case, structural) if mooring_provider is not None else None
        if mooring_terms is not None:
            if mooring_terms.stiffness.shape != effective_stiffness.shape:
                raise ValueError("mooring reduced stiffness shape must match reduced system")
            if mooring_terms.damping.shape != effective_damping.shape:
                raise ValueError("mooring reduced damping shape must match reduced system")
            if mooring_terms.pretension.size != effective_mass.shape[0]:
                raise ValueError("mooring reduced pretension length must match reduced system")
            effective_stiffness = effective_stiffness + mooring_terms.stiffness
            effective_damping = effective_damping + mooring_terms.damping
        force, wave_elevation, component_amplitude, component_phase = _build_excitation_force(
            hydrodynamic,
            config,
            omega,
            time,
            effective_mass.shape[0],
        )
        if mooring_terms is not None and np.any(mooring_terms.pretension):
            force = force + mooring_terms.pretension.reshape(1, -1)
        solved = solve_linear_time_domain(
            effective_mass,
            effective_damping,
            effective_stiffness,
            force,
            time,
            radiation_irf=radiation_irf,
            radiation_convolution_rule=config.radiation_convolution_rule,
        )
        global_displacement = reconstruct_global_response(
            structural.transformation,
            solved.displacement.T,
            structural.master_dofs,
            structural.slave_dofs,
            reverse_master_order=structural.reverse_master_order_for_reconstruction,
        ).T
        return RodmTimeDomainResult(
            time=time,
            master_displacement=solved.displacement,
            master_velocity=solved.velocity,
            master_acceleration=solved.acceleration,
            global_displacement=global_displacement,
            memory_force=solved.memory_force,
            master_nodes=structural.master_nodes,
            omega=omega,
            radiation_model=config.radiation_model,
            radiation_irf_time=hydrodynamic.radiation_irf_time,
            radiation_irf=hydrodynamic.radiation_irf,
            added_mass_infinite=hydrodynamic.added_mass_infinite,
            residual_added_mass=hydrodynamic.residual_added_mass,
            residual_radiation_damping=hydrodynamic.residual_radiation_damping,
            excitation_model=config.excitation_model,
            excitation_force=force,
            wave_elevation=wave_elevation,
            wave_component_omega=hydrodynamic.omega_grid if component_amplitude is not None else None,
            wave_component_amplitude=component_amplitude,
            wave_component_phase=component_phase,
            mooring_reduced_stiffness=(
                mooring_terms.stiffness
                if mooring_terms is not None and np.any(mooring_terms.stiffness)
                else None
            ),
            mooring_reduced_damping=(
                mooring_terms.damping
                if mooring_terms is not None and np.any(mooring_terms.damping)
                else None
            ),
            mooring_reduced_pretension=(
                mooring_terms.pretension
                if mooring_terms is not None and np.any(mooring_terms.pretension)
                else None
            ),
            mooring_metadata=(
                mooring_terms.metadata
                if mooring_terms is not None
                else {"enabled": False}
            ),
        )
    finally:
        dataset.close()


def _build_excitation_force(
    hydrodynamic,
    config: TimeDomainSimulationConfig,
    selected_omega: float,
    time: np.ndarray,
    ndof: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return force history and optional wave metadata for the configured input."""

    if config.excitation_model == "regular_wave":
        force = harmonic_force_time_series(
            hydrodynamic.wave_force.reshape(-1),
            selected_omega,
            time,
            amplitude=config.wave_amplitude,
            phase_rad=config.phase_rad,
            ramp_time=config.ramp_time,
        )
        return force, None, None, None

    if config.excitation_model == "external_force":
        return (
            external_force_time_series(
                config.external_force_time,
                config.external_force,
                time,
                expected_dofs=ndof,
            ),
            None,
            None,
            None,
        )

    if config.excitation_model != "wave_spectrum":
        raise ValueError("unsupported excitation_model")
    if hydrodynamic.wave_force_series is None:
        raise RuntimeError("wave_spectrum excitation requires multi-frequency wave forces")

    peak_period = (
        2.0 * np.pi / selected_omega
        if config.peak_period is None
        else float(config.peak_period)
    )
    spectrum = wave_spectrum_density(
        hydrodynamic.omega_grid,
        spectrum_type=config.spectrum_type,
        significant_wave_height=config.significant_wave_height,
        peak_period=peak_period,
        gamma=config.peak_enhancement_factor,
    )
    amplitudes = spectral_wave_amplitudes(hydrodynamic.omega_grid, spectrum)
    phases = random_wave_phases(len(amplitudes), seed=config.spectrum_seed)
    force = spectral_wave_force_time_series(
        hydrodynamic.wave_force_series,
        hydrodynamic.omega_grid,
        time,
        amplitudes,
        phases,
        ramp_time=config.ramp_time,
    )
    wave_elevation = wave_elevation_time_series(
        hydrodynamic.omega_grid,
        amplitudes,
        phases,
        time,
        ramp_time=config.ramp_time,
    )
    return force, wave_elevation, amplitudes, phases
