"""Run the external WEC-Sim-like time-domain platform for the DM10 RODM case.

This script is intentionally an adapter-layer entry point. It reads the RODM
frequency-domain inputs, selects either direct Cummins convolution or ERA
state-space radiation, optionally accepts a linearized mooring provider, and
writes time histories plus validation figures without changing the RODM
frequency-domain solver path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import csv
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.core import load_case_config, write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.mooring import (  # noqa: E402
    build_mooring_provider_from_config,
    is_mooring_enabled,
)
from offshore_energy_sim.time_domain import TimeDomainSimulationConfig, zero_mean_rms  # noqa: E402
from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    MooringLinearization,
    WecSimLikeRadiationConfig,
    WecSimLikeTimeDomainResult,
    build_corner_mooring_reduced_stiffness,
    corner_node_ids_for_regular_grid,
    solve_rodm_wecsim_like_time_domain,
)
from run_time_domain_reference_case_300 import (  # noqa: E402
    build_default_case,
    centerline_heave_time,
    default_dm_fem_root,
    representative_columns,
    write_representative_csv,
)


DEFAULT_HYDRO_FILE = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_spectrum_dense_88_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "wecsim_like_platform_dm10"
DEFAULT_TARGET_OMEGA = 0.4157
DOF_LABELS = ("surge", "sway", "heave", "roll", "pitch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument(
        "--radiation-model",
        choices=("direct_convolution", "state_space", "both"),
        default="both",
    )
    parser.add_argument(
        "--excitation-model",
        choices=("regular_wave", "wave_spectrum"),
        default="wave_spectrum",
    )
    parser.add_argument("--wave-amplitude", type=float, default=1.0)
    parser.add_argument("--phase-rad", type=float, default=0.0)
    parser.add_argument("--ramp-cycles", type=float, default=5.0)
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--spectrum-seed", type=int, default=1)
    parser.add_argument("--cycles", type=float, default=40.0)
    parser.add_argument("--steps-per-cycle", type=int, default=50)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
    parser.add_argument("--state-order", type=int, default=240)
    parser.add_argument("--era-block-rows", type=int, default=55)
    parser.add_argument("--era-block-cols", type=int, default=55)
    parser.add_argument("--state-space-model-path", type=Path, default=None)
    parser.add_argument("--save-state-space-model-path", type=Path, default=None)
    parser.add_argument("--integrator", choices=("newmark", "rk4"), default="newmark")
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
    parser.add_argument("--mooring-grid-nodes-x", type=int, default=61)
    parser.add_argument("--mooring-grid-nodes-y", type=int, default=13)
    parser.add_argument("--mooring-corner-horizontal-stiffness", type=float, default=0.0)
    parser.add_argument("--mooring-corner-surge-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-sway-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-heave-stiffness", type=float, default=0.0)
    parser.add_argument(
        "--mooring-config",
        type=Path,
        default=None,
        help="Optional YAML file with a WEC-Sim-style linear mooring section.",
    )
    parser.add_argument("--save-arrays", action="store_true")
    return parser.parse_args()


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    error = np.linalg.norm(np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float))
    scale = max(np.linalg.norm(np.asarray(reference, dtype=float)), 1.0e-30)
    return float(error / scale)


def rms_norm(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.linalg.norm(matrix, axis=1) ** 2)))


def resolve_mooring_stiffnesses(args: argparse.Namespace) -> tuple[float, float, float]:
    horizontal = float(args.mooring_corner_horizontal_stiffness)
    surge = horizontal if args.mooring_corner_surge_stiffness is None else float(args.mooring_corner_surge_stiffness)
    sway = horizontal if args.mooring_corner_sway_stiffness is None else float(args.mooring_corner_sway_stiffness)
    heave = float(args.mooring_corner_heave_stiffness)
    if min(surge, sway, heave) < 0.0:
        raise ValueError("mooring stiffness values must be non-negative")
    return surge, sway, heave


def build_optional_corner_mooring_provider(args: argparse.Namespace):
    if args.mooring_config is not None:
        config = load_case_config(args.mooring_config)
        provider = build_mooring_provider_from_config(config)
        return provider, {
            "enabled": is_mooring_enabled(config),
            "interface": "mooring_config_provider",
            "config_path": args.mooring_config,
            "fallback_corner_springs_used": False,
        }

    surge, sway, heave = resolve_mooring_stiffnesses(args)
    enabled = any(value > 0.0 for value in (surge, sway, heave))
    corner_nodes = corner_node_ids_for_regular_grid(args.mooring_grid_nodes_x, args.mooring_grid_nodes_y)
    metadata = {
        "enabled": enabled,
        "interface": "linearized_reduced_stiffness_provider",
        "example_model": "four_corner_linear_springs",
        "corner_nodes_one_based": corner_nodes,
        "grid_nodes_x": int(args.mooring_grid_nodes_x),
        "grid_nodes_y": int(args.mooring_grid_nodes_y),
        "surge_stiffness_per_corner": surge,
        "sway_stiffness_per_corner": sway,
        "heave_stiffness_per_corner": heave,
    }
    if not enabled:
        return None, metadata

    def provider(case, structural):
        reduced = build_corner_mooring_reduced_stiffness(
            total_nodes=case.total_nodes,
            retained_dofs_per_node=case.retained_dofs_per_node,
            nodes_per_x=args.mooring_grid_nodes_x,
            nodes_per_y=args.mooring_grid_nodes_y,
            transformation=structural.transformation,
            master_dofs=structural.master_dofs,
            slave_dofs=structural.slave_dofs,
            reverse_master_order=structural.reverse_master_order_for_reconstruction,
            surge_stiffness=surge,
            sway_stiffness=sway,
            heave_stiffness=heave,
        )
        return MooringLinearization(
            reduced,
            metadata={
                **metadata,
                "reduced_stiffness_frobenius_norm": float(np.linalg.norm(reduced)),
                "reduced_stiffness_trace": float(np.trace(reduced)),
            },
        )

    return provider, metadata


def select_case_and_period(args: argparse.Namespace):
    data_root = default_dm_fem_root(args.data_root)
    case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    case = replace(case, hydrodynamic_dataset=hydro_path)
    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset, merge_complex=True)
    try:
        omega_grid = np.asarray(dataset.omega.values, dtype=float).reshape(-1)
        frequency_index = int(np.argmin(np.abs(omega_grid - args.target_omega)))
        selected_omega = float(omega_grid[frequency_index])
    finally:
        dataset.close()
    period = 2.0 * np.pi / selected_omega
    return replace(case, frequency_index=frequency_index), selected_omega, period


def build_time_config(args: argparse.Namespace, period: float) -> TimeDomainSimulationConfig:
    return TimeDomainSimulationConfig(
        time_step=period / args.steps_per_cycle,
        duration=args.cycles * period,
        excitation_model=args.excitation_model,
        wave_amplitude=args.wave_amplitude,
        phase_rad=args.phase_rad,
        ramp_time=args.ramp_cycles * period,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=period,
        peak_enhancement_factor=args.peak_enhancement_factor,
        spectrum_seed=args.spectrum_seed,
        radiation_model="direct_convolution",
        memory_duration=args.memory_cycles * period,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_convolution_rule=args.radiation_convolution_rule,
        radiation_residual_model=args.radiation_residual_model,
    )


def solve_selected_models(args: argparse.Namespace) -> tuple[dict[str, WecSimLikeTimeDomainResult], dict[str, object], float]:
    case, selected_omega, period = select_case_and_period(args)
    config = build_time_config(args, period)
    provider, mooring_metadata = build_optional_corner_mooring_provider(args)
    requested = (
        ("direct_convolution", "state_space")
        if args.radiation_model == "both"
        else (args.radiation_model,)
    )
    results: dict[str, WecSimLikeTimeDomainResult] = {}
    for model_name in requested:
        radiation = WecSimLikeRadiationConfig(
            model=model_name,
            state_space_order=args.state_order,
            era_block_rows=args.era_block_rows,
            era_block_cols=args.era_block_cols,
            state_space_model_path=args.state_space_model_path if model_name == "state_space" else None,
            save_state_space_model_path=args.save_state_space_model_path if model_name == "state_space" else None,
            integrator=args.integrator,
        )
        results[model_name] = solve_rodm_wecsim_like_time_domain(
            case,
            config,
            radiation=radiation,
            mooring_provider=provider,
        )
    case_metadata = {
        "case_id": case.case_id,
        "hydrodynamic_dataset": case.hydrodynamic_dataset,
        "frequency_index": int(case.frequency_index),
        "selected_omega_rad_s": selected_omega,
        "period_s": period,
        "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
        "structural_reduction_method": case.structural_reduction_method,
        "serep_ridge_relative_lambda": case.serep_ridge_relative_lambda,
        "retained_dofs_per_node": case.retained_dofs_per_node,
        "mooring_requested": mooring_metadata,
    }
    return results, case_metadata, period


def result_metrics(result: WecSimLikeTimeDomainResult) -> dict[str, object]:
    heave = centerline_heave_time(
        result.global_displacement,
        retained_dofs_per_node=5,
    )
    metrics: dict[str, object] = {
        "radiation_model": result.radiation_model,
        "integrator": result.integrator,
        "excitation_model": result.excitation_model,
        "time_samples": int(result.time.size),
        "duration_s": float(result.time[-1]),
        "time_step_s": float(result.time[1] - result.time[0]),
        "master_displacement_shape": result.master_displacement.shape,
        "global_displacement_shape": result.global_displacement.shape,
        "centerline_heave_shape": heave.shape,
        "master_displacement_rms_norm": rms_norm(result.master_displacement),
        "master_velocity_rms_norm": rms_norm(result.master_velocity),
        "memory_force_rms_norm": rms_norm(result.memory_force),
        "centerline_heave_zero_mean_rms_norm": rms_norm(heave - np.mean(heave, axis=0, keepdims=True)),
        "centerline_heave_rms_mean": float(np.mean(zero_mean_rms(heave, axis=0))),
        "mooring": result.mooring_metadata,
    }
    if result.wave_elevation is not None:
        eta = np.asarray(result.wave_elevation, dtype=float)
        metrics["wave_elevation_zero_mean_rms"] = float(zero_mean_rms(eta))
        metrics["wave_elevation_reconstructed_hs"] = float(4.0 * zero_mean_rms(eta))
    if result.state_space_model is not None:
        metrics["state_space_model"] = result.state_space_model.to_dict()
    if result.state_space_model_path is not None:
        metrics["state_space_model_path"] = result.state_space_model_path
    return metrics


def comparison_metrics(
    direct: WecSimLikeTimeDomainResult,
    state: WecSimLikeTimeDomainResult,
) -> dict[str, object]:
    direct_heave = centerline_heave_time(direct.global_displacement, retained_dofs_per_node=5)
    state_heave = centerline_heave_time(state.global_displacement, retained_dofs_per_node=5)
    return {
        "state_vs_direct_master_displacement_l2_relative_error": relative_l2(
            state.master_displacement,
            direct.master_displacement,
        ),
        "state_vs_direct_master_velocity_l2_relative_error": relative_l2(
            state.master_velocity,
            direct.master_velocity,
        ),
        "state_vs_direct_memory_force_l2_relative_error": relative_l2(
            state.memory_force,
            direct.memory_force,
        ),
        "state_vs_direct_global_displacement_l2_relative_error": relative_l2(
            state.global_displacement,
            direct.global_displacement,
        ),
        "state_vs_direct_centerline_heave_l2_relative_error": relative_l2(
            state_heave,
            direct_heave,
        ),
        "state_vs_direct_centerline_heave_rms_relative_error": relative_l2(
            zero_mean_rms(state_heave, axis=0),
            zero_mean_rms(direct_heave, axis=0),
        ),
    }


def write_arrays(output_root: Path, name: str, result: WecSimLikeTimeDomainResult) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    heave = centerline_heave_time(result.global_displacement, retained_dofs_per_node=5)
    paths = {
        "time": output_root / f"{name}_time.npy",
        "master_displacement": output_root / f"{name}_master_displacement.npy",
        "master_velocity": output_root / f"{name}_master_velocity.npy",
        "master_acceleration": output_root / f"{name}_master_acceleration.npy",
        "global_displacement": output_root / f"{name}_global_displacement.npy",
        "memory_force": output_root / f"{name}_memory_force.npy",
        "excitation_force": output_root / f"{name}_excitation_force.npy",
        "centerline_heave": output_root / f"{name}_centerline_heave.npy",
    }
    np.save(paths["time"], result.time)
    np.save(paths["master_displacement"], result.master_displacement)
    np.save(paths["master_velocity"], result.master_velocity)
    np.save(paths["master_acceleration"], result.master_acceleration)
    np.save(paths["global_displacement"], result.global_displacement)
    np.save(paths["memory_force"], result.memory_force)
    np.save(paths["excitation_force"], result.excitation_force)
    np.save(paths["centerline_heave"], heave)
    if result.wave_elevation is not None:
        paths["wave_elevation"] = output_root / f"{name}_wave_elevation.npy"
        np.save(paths["wave_elevation"], result.wave_elevation)
    if result.wave_component_omega is not None:
        paths["wave_component_omega"] = output_root / f"{name}_wave_component_omega.npy"
        np.save(paths["wave_component_omega"], result.wave_component_omega)
    if result.wave_component_amplitude is not None:
        paths["wave_component_amplitude"] = output_root / f"{name}_wave_component_amplitude.npy"
        np.save(paths["wave_component_amplitude"], result.wave_component_amplitude)
    if result.wave_component_phase is not None:
        paths["wave_component_phase"] = output_root / f"{name}_wave_component_phase.npy"
        np.save(paths["wave_component_phase"], result.wave_component_phase)
    write_representative_csv(
        output_root / f"{name}_centerline_representative_heave.csv",
        result.time,
        heave,
    )
    return paths


def save_mooring_outputs(output_root: Path, result: WecSimLikeTimeDomainResult) -> dict[str, object]:
    """Save reduced mooring terms from a representative radiation result."""

    output_root.mkdir(parents=True, exist_ok=True)
    metadata = dict(result.mooring_metadata or {"enabled": False})
    summary: dict[str, object] = {
        "enabled": bool(metadata.get("enabled", False)),
        "metadata": metadata,
        "convention": "F_moor = F0 - K*q - C*qdot",
        "linearized_model": True,
        "summary_path": output_root / "mooring_summary.json",
    }
    ndof = None
    if result.mooring_reduced_stiffness is not None:
        ndof = result.mooring_reduced_stiffness.shape[0]
    elif result.mooring_reduced_damping is not None:
        ndof = result.mooring_reduced_damping.shape[0]
    elif result.mooring_reduced_pretension is not None:
        ndof = result.mooring_reduced_pretension.size

    stiffness = result.mooring_reduced_stiffness
    damping = result.mooring_reduced_damping
    pretension = result.mooring_reduced_pretension
    if summary["enabled"] and ndof is not None:
        if stiffness is None:
            stiffness = np.zeros((ndof, ndof), dtype=float)
        if damping is None:
            damping = np.zeros((ndof, ndof), dtype=float)
        if pretension is None:
            pretension = np.zeros(ndof, dtype=float)

    if stiffness is not None:
        path = output_root / "mooring_reduced_stiffness.npy"
        np.save(path, stiffness)
        summary["reduced_stiffness_path"] = path
        summary["reduced_stiffness_shape"] = stiffness.shape
        summary["reduced_stiffness_frobenius_norm"] = float(np.linalg.norm(stiffness))
        summary["reduced_stiffness_trace"] = float(np.trace(stiffness))
    if damping is not None:
        path = output_root / "mooring_reduced_damping.npy"
        np.save(path, damping)
        summary["reduced_damping_path"] = path
        summary["reduced_damping_shape"] = damping.shape
        summary["reduced_damping_frobenius_norm"] = float(np.linalg.norm(damping))
        summary["reduced_damping_trace"] = float(np.trace(damping))
    if pretension is not None:
        path = output_root / "mooring_reduced_pretension.npy"
        np.save(path, pretension)
        summary["reduced_pretension_path"] = path
        summary["reduced_pretension_shape"] = pretension.shape
        summary["reduced_pretension_norm"] = float(np.linalg.norm(pretension))
    active_terms = {
        "stiffness": bool(stiffness is not None and np.any(stiffness)),
        "damping": bool(damping is not None and np.any(damping)),
        "pretension": bool(pretension is not None and np.any(pretension)),
    }
    summary["active_terms"] = active_terms
    summary["response_change_expected"] = bool(any(active_terms.values()))
    write_metrics_json(summary["summary_path"], summary)
    return summary


def plot_representative_heave(path: Path, result: WecSimLikeTimeDomainResult) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    heave = centerline_heave_time(result.global_displacement, retained_dofs_per_node=5)
    columns = representative_columns(heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for column, label, color in zip(columns, labels, colors):
        ax.plot(result.time, heave[:, column], linewidth=1.1, color=color, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heave displacement")
    ax.set_title(f"{result.radiation_model} representative centerline heave")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_memory_force(path: Path, results: dict[str, WecSimLikeTimeDomainResult]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    styles = {
        "direct_convolution": ("#111111", "-"),
        "state_space": ("#d62728", "--"),
    }
    for name, result in results.items():
        color, linestyle = styles.get(name, ("#1f77b4", "-"))
        ax.plot(
            result.time,
            np.linalg.norm(result.memory_force, axis=1),
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            label=name,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radiation-memory force norm")
    ax.set_title("WEC-Sim-like radiation memory")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_state_direct_comparison(
    path: Path,
    direct: WecSimLikeTimeDomainResult,
    state: WecSimLikeTimeDomainResult,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    direct_heave = centerline_heave_time(direct.global_displacement, retained_dofs_per_node=5)
    state_heave = centerline_heave_time(state.global_displacement, retained_dofs_per_node=5)
    columns = representative_columns(direct_heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.0), sharex=True)
    for ax, column, label in zip(axes, columns, labels):
        ax.plot(direct.time, direct_heave[:, column], color="#111111", linewidth=1.0, label="direct")
        ax.plot(state.time, state_heave[:, column], color="#d62728", linestyle="--", linewidth=1.0, label="state-space")
        ax.set_ylabel("heave")
        ax.set_title(label)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Direct Cummins vs ERA state-space centerline heave", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_centerline_rms_comparison(
    path: Path,
    direct: WecSimLikeTimeDomainResult,
    state: WecSimLikeTimeDomainResult,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    direct_heave = centerline_heave_time(direct.global_displacement, retained_dofs_per_node=5)
    state_heave = centerline_heave_time(state.global_displacement, retained_dofs_per_node=5)
    x = np.linspace(0.0, 1.0, direct_heave.shape[1])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(x, zero_mean_rms(direct_heave, axis=0), color="#111111", linewidth=1.5, label="direct")
    ax.plot(x, zero_mean_rms(state_heave, axis=0), color="#d62728", linestyle="--", linewidth=1.3, label="state-space")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Zero-mean heave RMS")
    ax.set_title("Centerline heave RMS comparison")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_summary_csv(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, model_metrics in metrics["models"].items():
        rows.append(
            {
                "model": model_name,
                "time_samples": model_metrics["time_samples"],
                "master_displacement_rms_norm": model_metrics["master_displacement_rms_norm"],
                "memory_force_rms_norm": model_metrics["memory_force_rms_norm"],
                "centerline_heave_rms_mean": model_metrics["centerline_heave_rms_mean"],
                "wave_hs": model_metrics.get("wave_elevation_reconstructed_hs"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# WEC-Sim-like time-domain platform run",
        "",
        f"- status: `{metrics['status']}`",
        f"- adapter_layer_only: `{metrics['adapter_layer_only']}`",
        f"- rodm_frequency_core_modified: `{metrics['rodm_frequency_core_modified']}`",
        f"- excitation_model: `{metrics['time_config']['excitation_model']}`",
        f"- radiation_models: `{', '.join(metrics['models'].keys())}`",
        f"- selected_omega_rad_s: `{metrics['case']['selected_omega_rad_s']}`",
        f"- duration_s: `{metrics['time_config']['duration_s']}`",
        f"- time_step_s: `{metrics['time_config']['time_step_s']}`",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{figure}`" for figure in metrics["figures"])
    if "comparison" in metrics:
        lines.extend(["", "## State-space comparison", ""])
        for key, value in metrics["comparison"].items():
            lines.append(f"- `{key}`: `{value}`")
    mooring = metrics.get("mooring_outputs", {})
    if mooring:
        lines.extend(["", "## Mooring", ""])
        lines.append(f"- enabled: `{mooring.get('enabled')}`")
        lines.append(f"- convention: `{mooring.get('convention')}`")
        lines.append(f"- linearized_model: `{mooring.get('linearized_model')}`")
        lines.append(f"- active_terms: `{mooring.get('active_terms')}`")
        lines.append(f"- response_change_expected: `{mooring.get('response_change_expected')}`")
        lines.append(f"- summary_path: `{mooring.get('summary_path')}`")
        for key in (
            "reduced_stiffness_path",
            "reduced_damping_path",
            "reduced_pretension_path",
            "reduced_stiffness_frobenius_norm",
            "reduced_damping_frobenius_norm",
            "reduced_pretension_norm",
        ):
            if key in mooring:
                lines.append(f"- {key}: `{mooring[key]}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    start = timer.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    results, case_metadata, period = solve_selected_models(args)
    figures = []
    array_paths = {}
    for name, result in results.items():
        if args.save_arrays:
            array_paths[name] = write_arrays(output_root / "arrays", name, result)
        figures.append(
            plot_representative_heave(
                output_root / "figures" / f"{name}_representative_centerline_heave.png",
                result,
            )
        )
    figures.append(plot_memory_force(output_root / "figures" / "radiation_memory_force_norm.png", results))
    model_metrics = {name: result_metrics(result) for name, result in results.items()}
    mooring_outputs = save_mooring_outputs(output_root, next(iter(results.values())))
    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "case": case_metadata,
        "time_config": {
            "excitation_model": args.excitation_model,
            "time_step_s": period / args.steps_per_cycle,
            "duration_s": args.cycles * period,
            "cycles": args.cycles,
            "steps_per_cycle": args.steps_per_cycle,
            "ramp_cycles": args.ramp_cycles,
            "memory_cycles": args.memory_cycles,
            "wave_amplitude": args.wave_amplitude,
            "phase_rad": args.phase_rad,
            "spectrum_type": args.spectrum_type,
            "significant_wave_height": args.significant_wave_height,
            "peak_period_s": period,
            "peak_enhancement_factor": args.peak_enhancement_factor,
            "spectrum_seed": args.spectrum_seed,
            "radiation_passivity_correction": args.radiation_passivity_correction,
            "radiation_convolution_rule": args.radiation_convolution_rule,
            "radiation_residual_model": args.radiation_residual_model,
        },
        "platform": {
            "radiation_request": args.radiation_model,
            "state_space_order": args.state_order,
            "era_block_rows": args.era_block_rows,
            "era_block_cols": args.era_block_cols,
            "state_space_model_path": args.state_space_model_path,
            "save_state_space_model_path": args.save_state_space_model_path,
            "integrator": args.integrator,
            "mooring_interface": "optional reduced linearization provider",
        },
        "models": model_metrics,
        "figures": figures,
        "array_paths": array_paths,
        "mooring_outputs": mooring_outputs,
    }
    if {"direct_convolution", "state_space"}.issubset(results):
        metrics["comparison"] = comparison_metrics(
            results["direct_convolution"],
            results["state_space"],
        )
        figures.extend(
            [
                plot_state_direct_comparison(
                    output_root / "figures" / "direct_vs_state_centerline_heave_time.png",
                    results["direct_convolution"],
                    results["state_space"],
                ),
                plot_centerline_rms_comparison(
                    output_root / "figures" / "direct_vs_state_centerline_heave_rms.png",
                    results["direct_convolution"],
                    results["state_space"],
                ),
            ]
        )
    metrics["figures"] = figures
    metrics["summary_csv"] = write_summary_csv(output_root / "wecsim_like_platform_summary.csv", metrics)
    metrics["elapsed_seconds"] = float(timer.perf_counter() - start)
    metrics_path = write_metrics_json(output_root / "wecsim_like_platform_metrics.json", metrics)
    report_path = write_report(output_root / "report.md", metrics)

    print("WEC-Sim-like time-domain platform run completed.")
    print(f"radiation_models: {', '.join(results.keys())}")
    print(f"integrator: {args.integrator}")
    print(f"excitation_model: {args.excitation_model}")
    print(f"selected_omega_rad_s: {case_metadata['selected_omega_rad_s']:.9g}")
    if "comparison" in metrics:
        comparison = metrics["comparison"]
        print(
            "state/direct displacement and heave RMS errors: "
            f"{comparison['state_vs_direct_master_displacement_l2_relative_error']:.6g} / "
            f"{comparison['state_vs_direct_centerline_heave_rms_relative_error']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
