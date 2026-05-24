"""Validate ERA state-space radiation in a full linear RODM response solve."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.structure import calculate_node_positions, prepare_structural_reduction  # noqa: E402
from offshore_energy_sim.time_domain import TimeDomainSimulationConfig, solve_linear_time_domain  # noqa: E402
from offshore_energy_sim.time_domain.solver import _build_excitation_force  # noqa: E402
from offshore_energy_sim.time_domain.rodm_hydrodynamics import (  # noqa: E402
    prepare_rodm_time_domain_hydrodynamic_terms,
)
from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    StateSpaceRadiationLinearSystem,
    build_corner_mooring_reduced_stiffness,
    corner_node_ids_for_regular_grid,
    fit_era_state_space_radiation,
    solve_state_space_radiation_linear_system,
)
from run_time_domain_reference_case_300 import (  # noqa: E402
    build_default_case,
    centerline_heave_time,
    default_dm_fem_root,
    representative_columns,
)
from validate_state_space_radiation import DEFAULT_HYDRO, DEFAULT_TARGET_OMEGA, relative_l2  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "state_space_response_dense88_era240"
DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--state-order", type=int, default=240)
    parser.add_argument("--era-block-rows", type=int, default=55)
    parser.add_argument("--era-block-cols", type=int, default=55)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=60)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
    parser.add_argument("--spectrum-seed", type=int, default=1)
    parser.add_argument("--mooring-grid-nodes-x", type=int, default=61)
    parser.add_argument("--mooring-grid-nodes-y", type=int, default=13)
    parser.add_argument("--mooring-corner-horizontal-stiffness", type=float, default=0.0)
    parser.add_argument("--mooring-corner-surge-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-sway-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-heave-stiffness", type=float, default=0.0)
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default="clip_negative_eigenvalues",
    )
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default="trapezoidal",
    )
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default="selected_frequency",
    )
    parser.add_argument("--save-arrays", action="store_true")
    return parser.parse_args()


def rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2, axis=0))


def remove_linear_trend(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return values with the best-fit constant and linear trend removed."""

    t = np.asarray(time, dtype=float).reshape(-1)
    matrix = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones_like(t), t - t[0]])
    coefficients, *_ = np.linalg.lstsq(design, matrix, rcond=None)
    return matrix - design @ coefficients, coefficients[1]


def zero_mean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - np.mean(array, axis=0, keepdims=True)


def resolve_mooring_stiffnesses(args: argparse.Namespace) -> tuple[float, float, float]:
    horizontal = float(args.mooring_corner_horizontal_stiffness)
    surge = horizontal if args.mooring_corner_surge_stiffness is None else float(args.mooring_corner_surge_stiffness)
    sway = horizontal if args.mooring_corner_sway_stiffness is None else float(args.mooring_corner_sway_stiffness)
    heave = float(args.mooring_corner_heave_stiffness)
    if min(surge, sway, heave) < 0.0:
        raise ValueError("mooring stiffness values must be non-negative")
    return surge, sway, heave


def dof_group_metrics(
    direct: np.ndarray,
    state: np.ndarray,
    *,
    retained_dofs_per_node: int,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for dof in range(retained_dofs_per_node):
        label = DOF_LABELS[dof] if dof < len(DOF_LABELS) else f"dof_{dof}"
        direct_group = direct[:, dof::retained_dofs_per_node]
        state_group = state[:, dof::retained_dofs_per_node]
        metrics[label] = {
            "l2_relative_error": relative_l2(state_group, direct_group),
            "rms_relative_error": relative_l2(rms(state_group), rms(direct_group)),
            "zero_mean_l2_relative_error": relative_l2(
                zero_mean(state_group),
                zero_mean(direct_group),
            ),
            "direct_group_rms_norm": float(np.sqrt(np.mean(np.linalg.norm(direct_group, axis=1) ** 2))),
            "state_group_rms_norm": float(np.sqrt(np.mean(np.linalg.norm(state_group, axis=1) ** 2))),
            "direct_group_abs_max": float(np.max(np.abs(direct_group))),
            "state_group_abs_max": float(np.max(np.abs(state_group))),
        }
    return metrics


def plot_norm_comparison(
    path: Path,
    time: np.ndarray,
    direct: np.ndarray,
    state: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    direct_norm = np.linalg.norm(direct, axis=1)
    state_norm = np.linalg.norm(state, axis=1)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(time, direct_norm, color="#111111", linewidth=1.2, label="direct convolution")
    ax.plot(time, state_norm, color="#d62728", linestyle="--", linewidth=1.1, label="state-space")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_error_norm(
    path: Path,
    time: np.ndarray,
    direct: np.ndarray,
    state: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    error = np.linalg.norm(state - direct, axis=1)
    reference = np.linalg.norm(direct, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    axes[0].plot(time, error, color="#d62728", linewidth=1.1)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].plot(time, error / np.maximum(reference, 1.0e-30), color="#1f77b4", linewidth=1.1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("relative norm")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_centerline_heave_rms(
    path: Path,
    direct_heave: np.ndarray,
    state_heave: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, direct_heave.shape[1])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(x, rms(direct_heave), color="#111111", linewidth=1.6, label="direct convolution")
    ax.plot(x, rms(state_heave), color="#d62728", linestyle="--", linewidth=1.4, label="state-space")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Centerline heave RMS")
    ax.set_title("State-space vs direct Cummins centerline heave RMS")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_representative_centerline_heave(
    path: Path,
    time: np.ndarray,
    direct_heave: np.ndarray,
    state_heave: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = representative_columns(direct_heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.0), sharex=True)
    for ax, column, label in zip(axes, columns, labels):
        ax.plot(time, direct_heave[:, column], color="#111111", linewidth=1.1, label="direct")
        ax.plot(time, state_heave[:, column], color="#d62728", linestyle="--", linewidth=1.0, label="state-space")
        ax.set_ylabel("heave")
        ax.set_title(label)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Representative centerline heave histories", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_detrended_norm_comparison(
    path: Path,
    time: np.ndarray,
    direct: np.ndarray,
    state: np.ndarray,
) -> Path:
    direct_detrended, _ = remove_linear_trend(time, direct)
    state_detrended, _ = remove_linear_trend(time, state)
    return plot_norm_comparison(
        path,
        time,
        direct_detrended,
        state_detrended,
        ylabel="Detrended displacement norm",
        title="RODM oscillatory response after linear-trend removal",
    )


def main() -> int:
    args = parse_args()
    start = timer.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file

    base_case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    dataset = open_hydrodynamic_dataset(hydro_path, merge_complex=True)
    try:
        omega_grid = np.asarray(dataset.omega.values, dtype=float).reshape(-1)
        frequency_index = int(np.argmin(np.abs(omega_grid - args.target_omega)))
        selected_omega = float(omega_grid[frequency_index])
        peak_period = 2.0 * np.pi / selected_omega
        case = replace(
            base_case,
            hydrodynamic_dataset=hydro_path,
            frequency_index=frequency_index,
        )
        config = TimeDomainSimulationConfig(
            time_step=peak_period / args.steps_per_peak_cycle,
            duration=args.peak_cycles * peak_period,
            excitation_model="wave_spectrum",
            significant_wave_height=args.significant_wave_height,
            spectrum_type=args.spectrum_type,
            peak_period=peak_period,
            peak_enhancement_factor=args.peak_enhancement_factor,
            spectrum_seed=args.spectrum_seed,
            radiation_model="direct_convolution",
            memory_duration=args.memory_cycles * peak_period,
            radiation_passivity_correction=args.radiation_passivity_correction,
            radiation_convolution_rule=args.radiation_convolution_rule,
            radiation_residual_model=args.radiation_residual_model,
        )
        if case.master_nodes_one_based is None:
            master_nodes = calculate_node_positions(
                case.master_node_rule.first_node,
                case.master_node_rule.node_interval,
                case.master_node_rule.count,
            )
        else:
            master_nodes = list(case.master_nodes_one_based)
        structural = prepare_structural_reduction(case, master_nodes)
        hydrodynamic = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)
    finally:
        dataset.close()
    if hydrodynamic.added_mass_infinite is None or hydrodynamic.radiation_irf is None or hydrodynamic.radiation_irf_time is None:
        raise RuntimeError("state-space response validation requires direct-convolution hydrodynamic terms")

    mass_residual = (
        np.zeros_like(hydrodynamic.added_mass_infinite)
        if hydrodynamic.residual_added_mass is None
        else hydrodynamic.residual_added_mass
    )
    damping_residual = (
        np.zeros_like(hydrodynamic.radiation_damping)
        if hydrodynamic.residual_radiation_damping is None
        else hydrodynamic.residual_radiation_damping
    )
    effective_mass = structural.reduced_mass + hydrodynamic.added_mass_infinite + mass_residual
    effective_damping = damping_residual
    effective_stiffness = structural.reduced_stiffness + hydrodynamic.hydrostatic_stiffness
    mooring_surge, mooring_sway, mooring_heave = resolve_mooring_stiffnesses(args)
    mooring_enabled = any(value > 0.0 for value in (mooring_surge, mooring_sway, mooring_heave))
    mooring_reduced = np.zeros_like(effective_stiffness)
    mooring_corner_nodes = corner_node_ids_for_regular_grid(
        args.mooring_grid_nodes_x,
        args.mooring_grid_nodes_y,
    )
    if mooring_enabled:
        mooring_reduced = build_corner_mooring_reduced_stiffness(
            total_nodes=case.total_nodes,
            retained_dofs_per_node=case.retained_dofs_per_node,
            nodes_per_x=args.mooring_grid_nodes_x,
            nodes_per_y=args.mooring_grid_nodes_y,
            transformation=structural.transformation,
            master_dofs=structural.master_dofs,
            slave_dofs=structural.slave_dofs,
            reverse_master_order=structural.reverse_master_order_for_reconstruction,
            surge_stiffness=mooring_surge,
            sway_stiffness=mooring_sway,
            heave_stiffness=mooring_heave,
        )
        effective_stiffness = effective_stiffness + mooring_reduced
    time_values = config.time_values()
    force, _, _, _ = _build_excitation_force(
        hydrodynamic,
        config,
        selected_omega,
        time_values,
        effective_mass.shape[0],
    )
    direct = solve_linear_time_domain(
        effective_mass,
        effective_damping,
        effective_stiffness,
        force,
        time_values,
        radiation_irf=hydrodynamic.radiation_irf,
        radiation_convolution_rule=args.radiation_convolution_rule,
    )
    direct_global = reconstruct_global_response(
        structural.transformation,
        direct.displacement.T,
        structural.master_dofs,
        structural.slave_dofs,
        reverse_master_order=structural.reverse_master_order_for_reconstruction,
    ).T
    model = fit_era_state_space_radiation(
        hydrodynamic.radiation_irf_time,
        hydrodynamic.radiation_irf,
        order=args.state_order,
        block_rows=args.era_block_rows,
        block_cols=args.era_block_cols,
    )
    state = solve_state_space_radiation_linear_system(
        StateSpaceRadiationLinearSystem(
            mass=effective_mass,
            damping=effective_damping,
            stiffness=effective_stiffness,
            force=force,
            time=time_values,
            radiation_model=model,
        ),
        radiation_convolution_rule=args.radiation_convolution_rule,
    )
    state_global = reconstruct_global_response(
        structural.transformation,
        state.displacement.T,
        structural.master_dofs,
        structural.slave_dofs,
        reverse_master_order=structural.reverse_master_order_for_reconstruction,
    ).T
    direct_displacement_detrended, direct_displacement_slope = remove_linear_trend(
        direct.time,
        direct.displacement,
    )
    state_displacement_detrended, state_displacement_slope = remove_linear_trend(
        direct.time,
        state.displacement,
    )
    direct_centerline_heave = centerline_heave_time(
        direct_global,
        retained_dofs_per_node=case.retained_dofs_per_node,
    )
    state_centerline_heave = centerline_heave_time(
        state_global,
        retained_dofs_per_node=case.retained_dofs_per_node,
    )

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "hydrodynamic_dataset": hydro_path,
        "selected_omega_rad_s": selected_omega,
        "state_space_order": int(model.order),
        "era_block_rows": int(args.era_block_rows),
        "era_block_cols": int(args.era_block_cols),
        "state_space_model": model.to_dict(),
        "time_samples": int(direct.time.size),
        "duration_s": float(direct.time[-1]),
        "time_step_s": float(config.time_step),
        "memory_cycles": float(args.memory_cycles),
        "steps_per_peak_cycle": int(args.steps_per_peak_cycle),
        "radiation_convolution_rule": config.radiation_convolution_rule,
        "radiation_residual_model": config.radiation_residual_model,
        "mooring": {
            "enabled": mooring_enabled,
            "model": "four_corner_linear_springs",
            "corner_nodes_one_based": mooring_corner_nodes,
            "grid_nodes_x": int(args.mooring_grid_nodes_x),
            "grid_nodes_y": int(args.mooring_grid_nodes_y),
            "surge_stiffness_per_corner": mooring_surge,
            "sway_stiffness_per_corner": mooring_sway,
            "heave_stiffness_per_corner": mooring_heave,
            "reduced_stiffness_frobenius_norm": float(np.linalg.norm(mooring_reduced)),
            "reduced_stiffness_trace": float(np.trace(mooring_reduced)),
        },
        "master_displacement_l2_relative_error": relative_l2(
            state.displacement,
            direct.displacement,
        ),
        "master_velocity_l2_relative_error": relative_l2(
            state.velocity,
            direct.velocity,
        ),
        "master_acceleration_l2_relative_error": relative_l2(
            state.acceleration,
            direct.acceleration,
        ),
        "memory_force_l2_relative_error": relative_l2(
            state.memory_force,
            direct.memory_force,
        ),
        "global_displacement_l2_relative_error": relative_l2(
            state_global,
            direct_global,
        ),
        "master_displacement_rms_relative_error": relative_l2(
            rms(state.displacement),
            rms(direct.displacement),
        ),
        "master_displacement_detrended_l2_relative_error": relative_l2(
            state_displacement_detrended,
            direct_displacement_detrended,
        ),
        "master_displacement_drift_slope_relative_error": relative_l2(
            state_displacement_slope,
            direct_displacement_slope,
        ),
        "memory_force_rms_relative_error": relative_l2(
            rms(state.memory_force),
            rms(direct.memory_force),
        ),
        "master_dof_group_errors": dof_group_metrics(
            direct.displacement,
            state.displacement,
            retained_dofs_per_node=case.retained_dofs_per_node,
        ),
        "memory_force_dof_group_errors": dof_group_metrics(
            direct.memory_force,
            state.memory_force,
            retained_dofs_per_node=case.retained_dofs_per_node,
        ),
        "centerline_heave_l2_relative_error": relative_l2(
            state_centerline_heave,
            direct_centerline_heave,
        ),
        "centerline_heave_rms_relative_error": relative_l2(
            rms(state_centerline_heave),
            rms(direct_centerline_heave),
        ),
        "centerline_heave_zero_mean_l2_relative_error": relative_l2(
            zero_mean(state_centerline_heave),
            zero_mean(direct_centerline_heave),
        ),
        "centerline_heave_zero_mean_rms_relative_error": relative_l2(
            rms(zero_mean(state_centerline_heave)),
            rms(zero_mean(direct_centerline_heave)),
        ),
        "direct_master_displacement_rms": float(np.sqrt(np.mean(np.linalg.norm(direct.displacement, axis=1) ** 2))),
        "state_master_displacement_rms": float(np.sqrt(np.mean(np.linalg.norm(state.displacement, axis=1) ** 2))),
        "direct_centerline_heave_rms": float(np.sqrt(np.mean(np.linalg.norm(direct_centerline_heave, axis=1) ** 2))),
        "state_centerline_heave_rms": float(np.sqrt(np.mean(np.linalg.norm(state_centerline_heave, axis=1) ** 2))),
        "direct_memory_force_rms": float(np.sqrt(np.mean(np.linalg.norm(direct.memory_force, axis=1) ** 2))),
        "state_memory_force_rms": float(np.sqrt(np.mean(np.linalg.norm(state.memory_force, axis=1) ** 2))),
    }
    figures = [
        plot_norm_comparison(
            output_root / "figures" / "state_space_vs_direct_master_displacement_norm.png",
            direct.time,
            direct.displacement,
            state.displacement,
            ylabel="Master displacement norm",
            title="RODM response: state-space vs direct Cummins convolution",
        ),
        plot_norm_comparison(
            output_root / "figures" / "state_space_vs_direct_memory_force_norm.png",
            direct.time,
            direct.memory_force,
            state.memory_force,
            ylabel="Radiation-memory force norm",
            title="Radiation memory: state-space vs direct Cummins convolution",
        ),
        plot_error_norm(
            output_root / "figures" / "state_space_response_error_norm.png",
            direct.time,
            direct.displacement,
            state.displacement,
            ylabel="Displacement error norm",
            title="State-space response error",
        ),
        plot_detrended_norm_comparison(
            output_root / "figures" / "state_space_vs_direct_detrended_displacement_norm.png",
            direct.time,
            direct.displacement,
            state.displacement,
        ),
        plot_centerline_heave_rms(
            output_root / "figures" / "state_space_vs_direct_centerline_heave_rms.png",
            direct_centerline_heave,
            state_centerline_heave,
        ),
        plot_representative_centerline_heave(
            output_root / "figures" / "state_space_vs_direct_centerline_heave_time.png",
            direct.time,
            direct_centerline_heave,
            state_centerline_heave,
        ),
    ]
    if args.save_arrays:
        np.save(output_root / "state_space_master_displacement.npy", state.displacement)
        np.save(output_root / "state_space_master_velocity.npy", state.velocity)
        np.save(output_root / "state_space_memory_force.npy", state.memory_force)
        np.save(output_root / "state_space_global_displacement.npy", state_global)
    metrics["figures"] = figures
    metrics["elapsed_seconds"] = float(timer.perf_counter() - start)
    metrics_path = write_metrics_json(output_root / "state_space_response_metrics.json", metrics)

    print("State-space response validation completed.")
    print(f"state_space_order: {model.order}")
    if mooring_enabled:
        print(
            "mooring: "
            f"surge={mooring_surge:.6g}, sway={mooring_sway:.6g}, "
            f"heave={mooring_heave:.6g}, corners={mooring_corner_nodes}"
        )
    print(
        "master displacement/velocity/memory errors: "
        f"{metrics['master_displacement_l2_relative_error']:.6g} / "
        f"{metrics['master_velocity_l2_relative_error']:.6g} / "
        f"{metrics['memory_force_l2_relative_error']:.6g}"
    )
    print(
        "detrended displacement / drift slope errors: "
        f"{metrics['master_displacement_detrended_l2_relative_error']:.6g} / "
        f"{metrics['master_displacement_drift_slope_relative_error']:.6g}"
    )
    print(
        "centerline heave / zero-mean heave errors: "
        f"{metrics['centerline_heave_l2_relative_error']:.6g} / "
        f"{metrics['centerline_heave_zero_mean_l2_relative_error']:.6g}"
    )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
