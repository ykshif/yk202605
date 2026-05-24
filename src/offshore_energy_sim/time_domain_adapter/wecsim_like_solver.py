"""WEC-Sim-like external time-domain adapter for RODM cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset
from offshore_energy_sim.mooring import ReducedMooringTerms
from offshore_energy_sim.response import reconstruct_global_response
from offshore_energy_sim.structure import StructuralReductionResult, calculate_node_positions, prepare_structural_reduction
from offshore_energy_sim.time_domain import TimeDomainHydrodynamicTerms, TimeDomainSimulationConfig, solve_linear_time_domain
from offshore_energy_sim.time_domain.rodm_hydrodynamics import prepare_rodm_time_domain_hydrodynamic_terms
from offshore_energy_sim.time_domain.solver import _build_excitation_force
from offshore_energy_sim.time_domain_adapter.state_space_radiation import (
    DiscreteStateSpaceRadiationModel,
    fit_era_state_space_radiation,
    load_discrete_state_space_radiation_model,
    save_discrete_state_space_radiation_model,
)
from offshore_energy_sim.time_domain_adapter.state_space_solver import (
    StateSpaceRadiationLinearSystem,
    solve_state_space_radiation_linear_system,
)


MooringProvider = Callable[
    [RodmFrequencyCase, StructuralReductionResult],
    "MooringLinearization | ReducedMooringTerms | np.ndarray | None",
]


@dataclass(frozen=True)
class MooringLinearization:
    """Reduced linear mooring terms supplied by an external mooring module.

    The positional ``reduced_stiffness`` and ``metadata`` fields preserve the
    original adapter interface.  ``reduced_damping`` and ``reduced_pretension``
    extend it to the WEC-Sim Mooring Matrix convention:
    ``F_moor = F0 - K*q - C*qdot``.
    """

    reduced_stiffness: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)
    reduced_damping: np.ndarray | None = None
    reduced_pretension: np.ndarray | None = None


@dataclass(frozen=True)
class ResolvedMooringLinearization:
    """Validated reduced mooring terms used by the time-domain adapter."""

    reduced_stiffness: np.ndarray
    reduced_damping: np.ndarray
    reduced_pretension: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WecSimLikeRadiationConfig:
    """Radiation-model controls for the external WEC-Sim-like adapter."""

    model: str = "direct_convolution"
    state_space_order: int = 240
    era_block_rows: int | None = 55
    era_block_cols: int | None = 55
    state_space_model_path: Path | None = None
    save_state_space_model_path: Path | None = None
    integrator: str = "newmark"

    def __post_init__(self) -> None:
        model = str(self.model).lower()
        object.__setattr__(self, "model", model)
        integrator = str(self.integrator).lower()
        object.__setattr__(self, "integrator", integrator)
        if model not in {"direct_convolution", "state_space"}:
            raise ValueError("radiation model must be 'direct_convolution' or 'state_space'")
        if integrator not in {"newmark", "rk4"}:
            raise ValueError("integrator must be 'newmark' or 'rk4'")
        if self.state_space_order < 1:
            raise ValueError("state_space_order must be positive")
        if self.era_block_rows is not None and self.era_block_rows < 2:
            raise ValueError("era_block_rows must be at least 2")
        if self.era_block_cols is not None and self.era_block_cols < 2:
            raise ValueError("era_block_cols must be at least 2")


@dataclass(frozen=True)
class WecSimLikeTimeDomainResult:
    """Time histories from the external WEC-Sim-like adapter."""

    time: np.ndarray
    master_displacement: np.ndarray
    master_velocity: np.ndarray
    master_acceleration: np.ndarray
    global_displacement: np.ndarray
    memory_force: np.ndarray
    master_nodes: list[int]
    omega: float
    radiation_model: str
    integrator: str
    excitation_model: str
    excitation_force: np.ndarray
    wave_elevation: np.ndarray | None
    wave_component_omega: np.ndarray | None
    wave_component_amplitude: np.ndarray | None
    wave_component_phase: np.ndarray | None
    radiation_irf_time: np.ndarray
    radiation_irf: np.ndarray
    added_mass_infinite: np.ndarray
    residual_added_mass: np.ndarray | None
    residual_radiation_damping: np.ndarray | None
    state_space_model: DiscreteStateSpaceRadiationModel | None = None
    state_space_model_path: Path | None = None
    mooring_reduced_stiffness: np.ndarray | None = None
    mooring_reduced_damping: np.ndarray | None = None
    mooring_reduced_pretension: np.ndarray | None = None
    mooring_metadata: Mapping[str, object] = field(default_factory=dict)


def _master_nodes(case: RodmFrequencyCase) -> list[int]:
    if case.master_nodes_one_based is not None:
        return list(case.master_nodes_one_based)
    return calculate_node_positions(
        case.master_node_rule.first_node,
        case.master_node_rule.node_interval,
        case.master_node_rule.count,
    )


def _resolve_mooring(
    mooring: MooringLinearization | ReducedMooringTerms | np.ndarray | None,
    provider: MooringProvider | None,
    case: RodmFrequencyCase,
    structural: StructuralReductionResult,
    ndof: int,
) -> ResolvedMooringLinearization:
    supplied = provider(case, structural) if provider is not None else mooring
    zero_matrix = np.zeros((ndof, ndof), dtype=float)
    zero_vector = np.zeros(ndof, dtype=float)
    if supplied is None:
        return ResolvedMooringLinearization(
            zero_matrix,
            zero_matrix.copy(),
            zero_vector,
            {"enabled": False},
        )
    if isinstance(supplied, MooringLinearization):
        stiffness = _reduced_square_matrix(supplied.reduced_stiffness, ndof, "mooring reduced stiffness")
        damping = _reduced_square_matrix_or_zero(
            supplied.reduced_damping,
            ndof,
            "mooring reduced damping",
        )
        pretension = _reduced_vector_or_zero(
            supplied.reduced_pretension,
            ndof,
            "mooring reduced pretension",
        )
        metadata: Mapping[str, object] = supplied.metadata
    elif isinstance(supplied, ReducedMooringTerms):
        stiffness = _reduced_square_matrix(supplied.stiffness, ndof, "mooring reduced stiffness")
        damping = _reduced_square_matrix(supplied.damping, ndof, "mooring reduced damping")
        pretension = _reduced_vector_or_zero(
            supplied.pretension,
            ndof,
            "mooring reduced pretension",
        )
        metadata = supplied.metadata
    else:
        stiffness = _reduced_square_matrix(supplied, ndof, "mooring reduced stiffness")
        damping = zero_matrix.copy()
        pretension = zero_vector
        metadata = {"enabled": True, "source": "reduced_stiffness_matrix"}
    stiffness = 0.5 * (stiffness + stiffness.T)
    damping = 0.5 * (damping + damping.T)
    merged = {
        "enabled": bool(np.any(stiffness) or np.any(damping) or np.any(pretension)),
        **dict(metadata),
    }
    return ResolvedMooringLinearization(stiffness, damping, pretension, merged)


def _reduced_square_matrix(matrix: np.ndarray, ndof: int, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (ndof, ndof):
        raise ValueError(f"{name} must match reduced system DOFs")
    return values


def _reduced_square_matrix_or_zero(
    matrix: np.ndarray | None,
    ndof: int,
    name: str,
) -> np.ndarray:
    if matrix is None:
        return np.zeros((ndof, ndof), dtype=float)
    return _reduced_square_matrix(matrix, ndof, name)


def _reduced_vector_or_zero(
    vector: np.ndarray | None,
    ndof: int,
    name: str,
) -> np.ndarray:
    if vector is None:
        return np.zeros(ndof, dtype=float)
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != ndof:
        raise ValueError(f"{name} length must match reduced system DOFs")
    return values


def _fit_or_load_state_space_model(
    hydrodynamic: TimeDomainHydrodynamicTerms,
    radiation: WecSimLikeRadiationConfig,
) -> tuple[DiscreteStateSpaceRadiationModel, Path | None]:
    if radiation.state_space_model_path is not None:
        model = load_discrete_state_space_radiation_model(radiation.state_space_model_path)
        return model, Path(radiation.state_space_model_path)
    if hydrodynamic.radiation_irf is None or hydrodynamic.radiation_irf_time is None:
        raise RuntimeError("state_space radiation requires preprocessed Cummins IRF terms")
    model = fit_era_state_space_radiation(
        hydrodynamic.radiation_irf_time,
        hydrodynamic.radiation_irf,
        order=radiation.state_space_order,
        block_rows=radiation.era_block_rows,
        block_cols=radiation.era_block_cols,
    )
    saved_path = None
    if radiation.save_state_space_model_path is not None:
        saved_path = save_discrete_state_space_radiation_model(
            model,
            radiation.save_state_space_model_path,
        )
    return model, saved_path


def solve_rodm_wecsim_like_time_domain(
    case: RodmFrequencyCase,
    config: TimeDomainSimulationConfig,
    *,
    radiation: WecSimLikeRadiationConfig | None = None,
    mooring: MooringLinearization | ReducedMooringTerms | np.ndarray | None = None,
    mooring_provider: MooringProvider | None = None,
) -> WecSimLikeTimeDomainResult:
    """Run the external WEC-Sim-like time-domain adapter for one RODM case.

    The function reads RODM-compatible frequency-domain inputs, builds the
    Cummins terms in the adapter layer, optionally adds an external reduced
    mooring stiffness, and solves either direct Cummins convolution or an ERA
    state-space radiation realization.
    """

    radiation_config = WecSimLikeRadiationConfig() if radiation is None else radiation
    preprocessing_config = replace(config, radiation_model="direct_convolution")
    master_nodes = _master_nodes(case)
    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset, merge_complex=True)
    try:
        structural = prepare_structural_reduction(case, master_nodes)
        hydrodynamic = prepare_rodm_time_domain_hydrodynamic_terms(
            case,
            dataset,
            preprocessing_config,
        )
    finally:
        dataset.close()
    if (
        hydrodynamic.added_mass_infinite is None
        or hydrodynamic.radiation_irf is None
        or hydrodynamic.radiation_irf_time is None
    ):
        raise RuntimeError("WEC-Sim-like adapter requires Cummins radiation terms")

    residual_mass = (
        np.zeros_like(hydrodynamic.added_mass_infinite)
        if hydrodynamic.residual_added_mass is None
        else hydrodynamic.residual_added_mass
    )
    residual_damping = (
        np.zeros_like(hydrodynamic.radiation_damping)
        if hydrodynamic.residual_radiation_damping is None
        else hydrodynamic.residual_radiation_damping
    )
    mass = structural.reduced_mass + hydrodynamic.added_mass_infinite + residual_mass
    damping = residual_damping
    stiffness = structural.reduced_stiffness + hydrodynamic.hydrostatic_stiffness
    resolved_mooring = _resolve_mooring(
        mooring,
        mooring_provider,
        case,
        structural,
        mass.shape[0],
    )
    stiffness = stiffness + resolved_mooring.reduced_stiffness
    damping = damping + resolved_mooring.reduced_damping
    time = preprocessing_config.time_values()
    selected_omega = float(np.asarray(hydrodynamic.omega).reshape(-1)[0])
    force, wave_elevation, component_amplitude, component_phase = _build_excitation_force(
        hydrodynamic,
        preprocessing_config,
        selected_omega,
        time,
        mass.shape[0],
    )
    if np.any(resolved_mooring.reduced_pretension):
        force = force + resolved_mooring.reduced_pretension.reshape(1, -1)

    state_model = None
    state_model_path = None
    if radiation_config.model == "direct_convolution":
        solved = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            radiation_irf=hydrodynamic.radiation_irf,
            radiation_convolution_rule=preprocessing_config.radiation_convolution_rule,
            integrator=radiation_config.integrator,
        )
    else:
        state_model, state_model_path = _fit_or_load_state_space_model(hydrodynamic, radiation_config)
        solved = solve_state_space_radiation_linear_system(
            StateSpaceRadiationLinearSystem(
                mass=mass,
                damping=damping,
                stiffness=stiffness,
                force=force,
                time=time,
                radiation_model=state_model,
            ),
            radiation_convolution_rule=preprocessing_config.radiation_convolution_rule,
            integrator=radiation_config.integrator,
        )

    global_displacement = reconstruct_global_response(
        structural.transformation,
        solved.displacement.T,
        structural.master_dofs,
        structural.slave_dofs,
        reverse_master_order=structural.reverse_master_order_for_reconstruction,
    ).T
    return WecSimLikeTimeDomainResult(
        time=time,
        master_displacement=solved.displacement,
        master_velocity=solved.velocity,
        master_acceleration=solved.acceleration,
        global_displacement=global_displacement,
        memory_force=solved.memory_force,
        master_nodes=structural.master_nodes,
        omega=selected_omega,
        radiation_model=radiation_config.model,
        integrator=radiation_config.integrator,
        excitation_model=preprocessing_config.excitation_model,
        excitation_force=force,
        wave_elevation=wave_elevation,
        wave_component_omega=hydrodynamic.omega_grid if component_amplitude is not None else None,
        wave_component_amplitude=component_amplitude,
        wave_component_phase=component_phase,
        radiation_irf_time=hydrodynamic.radiation_irf_time,
        radiation_irf=hydrodynamic.radiation_irf,
        added_mass_infinite=hydrodynamic.added_mass_infinite,
        residual_added_mass=hydrodynamic.residual_added_mass,
        residual_radiation_damping=hydrodynamic.residual_radiation_damping,
        state_space_model=state_model,
        state_space_model_path=state_model_path,
        mooring_reduced_stiffness=(
            resolved_mooring.reduced_stiffness
            if np.any(resolved_mooring.reduced_stiffness)
            else None
        ),
        mooring_reduced_damping=(
            resolved_mooring.reduced_damping
            if np.any(resolved_mooring.reduced_damping)
            else None
        ),
        mooring_reduced_pretension=(
            resolved_mooring.reduced_pretension
            if np.any(resolved_mooring.reduced_pretension)
            else None
        ),
        mooring_metadata=resolved_mooring.metadata,
    )
