"""Run multi-sea-state validation for the external WEC-Sim-like adapter.

The script keeps the RODM frequency-domain core and the mooring module
untouched. It orchestrates repeated adapter solves, compares ERA state-space
radiation against direct Cummins convolution, and runs one longer lightweight
state-space case for practical time-series screening.
"""

from __future__ import annotations

from copy import deepcopy
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

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset, prepare_hydrodynamic_terms  # noqa: E402
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.structure import calculate_node_positions, prepare_structural_reduction  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_multi_harmonic_amplitudes,
    relative_l2_error,
    spectral_wave_amplitudes,
    wave_spectrum_density,
    zero_mean_rms,
)
from run_time_domain_reference_case_300 import build_default_case, centerline_heave_time, default_dm_fem_root  # noqa: E402
from run_wecsim_like_time_domain_platform import (  # noqa: E402
    DEFAULT_HYDRO_FILE,
    build_optional_corner_mooring_provider,
    comparison_metrics,
    result_metrics,
    solve_selected_models,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "wecsim_like_multi_sea_state_validation"


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hs-values", default="0.5,1.0")
    parser.add_argument("--target-omega-values", default="0.35,0.4157,0.55")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--cycles", type=float, default=20.0)
    parser.add_argument("--steps-per-cycle", type=int, default=40)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
    parser.add_argument("--long-cycles", type=float, default=120.0)
    parser.add_argument("--long-target-omega", type=float, default=0.4157)
    parser.add_argument("--long-hs", type=float, default=1.0)
    parser.add_argument("--long-seed", type=int, default=1)
    parser.add_argument("--skip-long-run", action="store_true")
    parser.add_argument("--skip-frequency-rms", action="store_true")
    parser.add_argument("--frequency-rms-discard-cycles", type=float, default=5.0)
    parser.add_argument(
        "--fit-min-samples-per-parameter",
        type=float,
        default=8.0,
        help="Only run multi-harmonic fits when enough post-discard samples are available.",
    )
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--state-order", type=int, default=240)
    parser.add_argument("--era-block-rows", type=int, default=55)
    parser.add_argument("--era-block-cols", type=int, default=55)
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
    parser.add_argument("--mooring-corner-horizontal-stiffness", type=float, default=1.0e7)
    parser.add_argument("--mooring-corner-surge-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-sway-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-heave-stiffness", type=float, default=0.0)
    return parser.parse_args()


def build_platform_args(
    args: argparse.Namespace,
    *,
    output_root: Path,
    target_omega: float,
    hs: float,
    seed: int,
    cycles: float,
    radiation_model: str,
) -> argparse.Namespace:
    platform_args = argparse.Namespace(
        data_root=args.data_root,
        hydro_file=args.hydro_file,
        output_root=output_root,
        target_omega=float(target_omega),
        hydro_node_order=args.hydro_node_order,
        radiation_model=radiation_model,
        excitation_model="wave_spectrum",
        wave_amplitude=1.0,
        phase_rad=0.0,
        ramp_cycles=5.0,
        significant_wave_height=float(hs),
        spectrum_type=args.spectrum_type,
        peak_enhancement_factor=args.peak_enhancement_factor,
        spectrum_seed=int(seed),
        cycles=float(cycles),
        steps_per_cycle=args.steps_per_cycle,
        memory_cycles=args.memory_cycles,
        state_order=args.state_order,
        era_block_rows=args.era_block_rows,
        era_block_cols=args.era_block_cols,
        integrator=args.integrator,
        state_space_model_path=None,
        save_state_space_model_path=None,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_convolution_rule=args.radiation_convolution_rule,
        radiation_residual_model=args.radiation_residual_model,
        mooring_grid_nodes_x=args.mooring_grid_nodes_x,
        mooring_grid_nodes_y=args.mooring_grid_nodes_y,
        mooring_corner_horizontal_stiffness=args.mooring_corner_horizontal_stiffness,
        mooring_corner_surge_stiffness=args.mooring_corner_surge_stiffness,
        mooring_corner_sway_stiffness=args.mooring_corner_sway_stiffness,
        mooring_corner_heave_stiffness=args.mooring_corner_heave_stiffness,
        save_arrays=False,
    )
    return platform_args


def case_id(hs: float, omega: float, seed: int) -> str:
    hs_tag = f"{hs:.3g}".replace(".", "p")
    omega_tag = f"{omega:.4g}".replace(".", "p")
    return f"Hs{hs_tag}_om{omega_tag}_seed{seed}"


def compact_model_metrics(model_metrics: dict[str, object]) -> dict[str, float]:
    keys = (
        "master_displacement_rms_norm",
        "master_velocity_rms_norm",
        "memory_force_rms_norm",
        "centerline_heave_zero_mean_rms_norm",
        "centerline_heave_rms_mean",
        "wave_elevation_reconstructed_hs",
    )
    return {key: float(model_metrics[key]) for key in keys if key in model_metrics}


def resolve_hydro_path(args: argparse.Namespace) -> Path:
    data_root = default_dm_fem_root(args.data_root)
    return args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file


def solve_frequency_centerline_rao(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve centerline heave RAOs on the full hydrodynamic omega grid."""

    data_root = default_dm_fem_root(args.data_root)
    hydro_path = resolve_hydro_path(args)
    case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    case = replace(case, hydrodynamic_dataset=hydro_path)
    if case.master_nodes_one_based is None:
        master_nodes = calculate_node_positions(
            case.master_node_rule.first_node,
            case.master_node_rule.node_interval,
            case.master_node_rule.count,
        )
    else:
        master_nodes = list(case.master_nodes_one_based)

    dataset = open_hydrodynamic_dataset(hydro_path, merge_complex=True)
    try:
        omega = np.asarray(dataset.omega.values, dtype=float).reshape(-1)
        structural = prepare_structural_reduction(case, master_nodes)
        provider, _ = build_optional_corner_mooring_provider(args)
        mooring_stiffness = np.zeros_like(structural.reduced_stiffness)
        if provider is not None:
            supplied = provider(case, structural)
            if supplied is not None:
                if hasattr(supplied, "reduced_stiffness"):
                    mooring_stiffness = np.asarray(supplied.reduced_stiffness, dtype=float)
                else:
                    mooring_stiffness = np.asarray(supplied, dtype=float)
                mooring_stiffness = 0.5 * (mooring_stiffness + mooring_stiffness.T)

        heave_rao = []
        for frequency_index in range(omega.size):
            frequency_case = replace(case, frequency_index=int(frequency_index))
            hydrodynamic = prepare_hydrodynamic_terms(frequency_case, dataset)
            effective_mass = hydrodynamic.added_mass + structural.reduced_mass
            effective_damping = hydrodynamic.radiation_damping
            effective_stiffness = (
                hydrodynamic.hydrostatic_stiffness
                + structural.reduced_stiffness
                + mooring_stiffness
            )
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
            heave_rao.append(
                centerline_heave_time(
                    global_displacement.reshape(1, -1),
                    retained_dofs_per_node=case.retained_dofs_per_node,
                )[0]
            )
    finally:
        dataset.close()
    return omega, np.stack(heave_rao, axis=0)


def frequency_centerline_rms(
    *,
    omega: np.ndarray,
    heave_rao: np.ndarray,
    spectrum_type: str,
    significant_wave_height: float,
    peak_period: float,
    peak_enhancement_factor: float,
) -> np.ndarray:
    spectrum = wave_spectrum_density(
        omega,
        spectrum_type=spectrum_type,
        significant_wave_height=significant_wave_height,
        peak_period=peak_period,
        gamma=peak_enhancement_factor,
    )
    amplitudes = spectral_wave_amplitudes(omega, spectrum)
    components = heave_rao * amplitudes[:, np.newaxis]
    return np.sqrt(0.5 * np.sum(np.abs(components) ** 2, axis=0))


def post_discard_heave_rms(
    heave: np.ndarray,
    time: np.ndarray,
    *,
    discard_seconds: float,
) -> np.ndarray:
    mask = np.asarray(time, dtype=float) >= float(time[0] + discard_seconds)
    if np.count_nonzero(mask) < 3:
        mask = np.ones_like(time, dtype=bool)
    return zero_mean_rms(heave[mask], axis=0)


def run_short_validation_case(
    args: argparse.Namespace,
    *,
    hs: float,
    target_omega: float,
    seed: int,
    output_root: Path,
    frequency_omega: np.ndarray | None = None,
    frequency_heave_rao: np.ndarray | None = None,
) -> dict[str, object]:
    start = timer.perf_counter()
    run_args = build_platform_args(
        args,
        output_root=output_root,
        target_omega=target_omega,
        hs=hs,
        seed=seed,
        cycles=args.cycles,
        radiation_model="both",
    )
    results, case_metadata, _ = solve_selected_models(run_args)
    direct = results["direct_convolution"]
    state = results["state_space"]
    comparison = comparison_metrics(direct, state)
    direct_metrics = result_metrics(direct)
    state_metrics = result_metrics(state)
    direct_heave = centerline_heave_time(direct.global_displacement, retained_dofs_per_node=5)
    state_heave = centerline_heave_time(state.global_displacement, retained_dofs_per_node=5)
    item = {
        "case_id": case_id(hs, target_omega, seed),
        "target_omega_rad_s": float(target_omega),
        "selected_omega_rad_s": float(case_metadata["selected_omega_rad_s"]),
        "peak_period_s": float(case_metadata["period_s"]),
        "significant_wave_height": float(hs),
        "spectrum_seed": int(seed),
        "duration_s": float(direct.time[-1]),
        "time_step_s": float(direct.time[1] - direct.time[0]),
        "time_samples": int(direct.time.size),
        "direct": compact_model_metrics(direct_metrics),
        "state_space": compact_model_metrics(state_metrics),
        "state_space_model": (
            deepcopy(state_metrics["state_space_model"])
            if "state_space_model" in state_metrics
            else None
        ),
        "comparison": comparison,
        "direct_centerline_heave_rms": zero_mean_rms(direct_heave, axis=0),
        "state_centerline_heave_rms": zero_mean_rms(state_heave, axis=0),
        "elapsed_seconds": float(timer.perf_counter() - start),
    }
    if frequency_omega is not None and frequency_heave_rao is not None:
        frequency_rms = frequency_centerline_rms(
            omega=frequency_omega,
            heave_rao=frequency_heave_rao,
            spectrum_type=args.spectrum_type,
            significant_wave_height=float(hs),
            peak_period=float(case_metadata["period_s"]),
            peak_enhancement_factor=args.peak_enhancement_factor,
        )
        discard_seconds = args.frequency_rms_discard_cycles * float(case_metadata["period_s"])
        direct_post_rms = post_discard_heave_rms(
            direct_heave,
            direct.time,
            discard_seconds=discard_seconds,
        )
        state_post_rms = post_discard_heave_rms(
            state_heave,
            state.time,
            discard_seconds=discard_seconds,
        )
        item["frequency_rms"] = {
            "discard_seconds": float(discard_seconds),
            "frequency_centerline_heave_rms_mean": float(np.mean(frequency_rms)),
            "direct_post_discard_heave_rms_mean": float(np.mean(direct_post_rms)),
            "state_post_discard_heave_rms_mean": float(np.mean(state_post_rms)),
            "frequency_vs_direct_post_discard_rms_l2_relative_error": relative_l2_error(
                direct_post_rms,
                frequency_rms,
            ),
            "frequency_vs_state_post_discard_rms_l2_relative_error": relative_l2_error(
                state_post_rms,
                frequency_rms,
            ),
            "frequency_vs_direct_fit_rms_l2_relative_error": None,
            "frequency_vs_state_fit_rms_l2_relative_error": None,
            "direct_fit_heave_rms_mean": None,
            "state_fit_heave_rms_mean": None,
            "state_vs_frequency_centerline_heave_rms": state_post_rms,
            "direct_vs_frequency_centerline_heave_rms": direct_post_rms,
            "frequency_centerline_heave_rms": frequency_rms,
        }
        post_sample_count = int(np.count_nonzero(direct.time >= direct.time[0] + discard_seconds))
        fit_parameter_count = 2 * frequency_omega.size
        if post_sample_count > args.fit_min_samples_per_parameter * fit_parameter_count:
            direct_components = fit_multi_harmonic_amplitudes(
                direct_heave,
                direct.time,
                frequency_omega,
                start_time=direct.time[0] + discard_seconds,
            )
            state_components = fit_multi_harmonic_amplitudes(
                state_heave,
                state.time,
                frequency_omega,
                start_time=state.time[0] + discard_seconds,
            )
            direct_fit_rms = np.sqrt(0.5 * np.sum(np.abs(direct_components) ** 2, axis=0))
            state_fit_rms = np.sqrt(0.5 * np.sum(np.abs(state_components) ** 2, axis=0))
            item["frequency_rms"].update(
                {
                    "fit_status": "completed",
                    "frequency_vs_direct_fit_rms_l2_relative_error": relative_l2_error(
                        direct_fit_rms,
                        frequency_rms,
                    ),
                    "frequency_vs_state_fit_rms_l2_relative_error": relative_l2_error(
                        state_fit_rms,
                        frequency_rms,
                    ),
                    "direct_fit_heave_rms_mean": float(np.mean(direct_fit_rms)),
                    "state_fit_heave_rms_mean": float(np.mean(state_fit_rms)),
                    "direct_fit_centerline_heave_rms": direct_fit_rms,
                    "state_fit_centerline_heave_rms": state_fit_rms,
                }
            )
        else:
            item["frequency_rms"]["fit_status"] = "skipped_insufficient_post_discard_samples"
            item["frequency_rms"]["fit_post_sample_count"] = post_sample_count
            item["frequency_rms"]["fit_parameter_count"] = fit_parameter_count
    return item


def windowed_rms(values: np.ndarray, window_count: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape[0] < window_count:
        raise ValueError("not enough samples for requested window_count")
    windows = np.array_split(matrix, window_count, axis=0)
    return np.array([zero_mean_rms(window, axis=0) for window in windows])


def run_long_lightweight_case(args: argparse.Namespace, output_root: Path) -> dict[str, object]:
    start = timer.perf_counter()
    run_args = build_platform_args(
        args,
        output_root=output_root,
        target_omega=args.long_target_omega,
        hs=args.long_hs,
        seed=args.long_seed,
        cycles=args.long_cycles,
        radiation_model="state_space",
    )
    results, case_metadata, _ = solve_selected_models(run_args)
    result = results["state_space"]
    heave = centerline_heave_time(result.global_displacement, retained_dofs_per_node=5)
    heave_window_rms = windowed_rms(heave, 6)
    wave_window_hs = None
    if result.wave_elevation is not None:
        wave_window_hs = 4.0 * windowed_rms(result.wave_elevation.reshape(-1, 1), 6).reshape(-1)
    return {
        "case_id": case_id(args.long_hs, args.long_target_omega, args.long_seed),
        "target_omega_rad_s": float(args.long_target_omega),
        "selected_omega_rad_s": float(case_metadata["selected_omega_rad_s"]),
        "peak_period_s": float(case_metadata["period_s"]),
        "significant_wave_height": float(args.long_hs),
        "spectrum_seed": int(args.long_seed),
        "duration_s": float(result.time[-1]),
        "time_step_s": float(result.time[1] - result.time[0]),
        "time_samples": int(result.time.size),
        "state_space": compact_model_metrics(result_metrics(result)),
        "centerline_heave_window_rms_mean": np.mean(heave_window_rms, axis=1),
        "centerline_heave_window_rms_max": np.max(heave_window_rms, axis=1),
        "wave_window_hs": wave_window_hs,
        "elapsed_seconds": float(timer.perf_counter() - start),
        "lightweight_output": "metrics_and_figures_only_no_large_arrays_saved",
    }


def write_summary_csv(path: Path, cases: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "Hs",
        "target_omega",
        "selected_omega",
        "Tp",
        "seed",
        "samples",
        "duration_s",
        "state_master_error",
        "state_memory_error",
        "state_heave_error",
        "state_heave_rms_error",
        "direct_heave_rms_mean",
        "state_heave_rms_mean",
        "wave_hs_reconstructed",
        "era_fit_error",
        "frequency_heave_rms_mean",
        "direct_post_discard_heave_rms_mean",
        "state_post_discard_heave_rms_mean",
        "frequency_vs_direct_post_discard_rms_error",
        "frequency_vs_state_post_discard_rms_error",
        "frequency_vs_direct_fit_rms_error",
        "frequency_vs_state_fit_rms_error",
        "direct_fit_heave_rms_mean",
        "state_fit_heave_rms_mean",
        "fit_status",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in cases:
            model = item.get("state_space_model") or {}
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "Hs": item["significant_wave_height"],
                    "target_omega": item["target_omega_rad_s"],
                    "selected_omega": item["selected_omega_rad_s"],
                    "Tp": item["peak_period_s"],
                    "seed": item["spectrum_seed"],
                    "samples": item["time_samples"],
                    "duration_s": item["duration_s"],
                    "state_master_error": item["comparison"]["state_vs_direct_master_displacement_l2_relative_error"],
                    "state_memory_error": item["comparison"]["state_vs_direct_memory_force_l2_relative_error"],
                    "state_heave_error": item["comparison"]["state_vs_direct_centerline_heave_l2_relative_error"],
                    "state_heave_rms_error": item["comparison"]["state_vs_direct_centerline_heave_rms_relative_error"],
                    "direct_heave_rms_mean": item["direct"]["centerline_heave_rms_mean"],
                    "state_heave_rms_mean": item["state_space"]["centerline_heave_rms_mean"],
                    "wave_hs_reconstructed": item["direct"].get("wave_elevation_reconstructed_hs"),
                    "era_fit_error": model.get("fit_l2_relative_error"),
                    "frequency_heave_rms_mean": (item.get("frequency_rms") or {}).get("frequency_centerline_heave_rms_mean"),
                    "direct_post_discard_heave_rms_mean": (item.get("frequency_rms") or {}).get("direct_post_discard_heave_rms_mean"),
                    "state_post_discard_heave_rms_mean": (item.get("frequency_rms") or {}).get("state_post_discard_heave_rms_mean"),
                    "frequency_vs_direct_post_discard_rms_error": (item.get("frequency_rms") or {}).get(
                        "frequency_vs_direct_post_discard_rms_l2_relative_error"
                    ),
                    "frequency_vs_state_post_discard_rms_error": (item.get("frequency_rms") or {}).get(
                        "frequency_vs_state_post_discard_rms_l2_relative_error"
                    ),
                    "frequency_vs_direct_fit_rms_error": (item.get("frequency_rms") or {}).get(
                        "frequency_vs_direct_fit_rms_l2_relative_error"
                    ),
                    "frequency_vs_state_fit_rms_error": (item.get("frequency_rms") or {}).get(
                        "frequency_vs_state_fit_rms_l2_relative_error"
                    ),
                    "direct_fit_heave_rms_mean": (item.get("frequency_rms") or {}).get("direct_fit_heave_rms_mean"),
                    "state_fit_heave_rms_mean": (item.get("frequency_rms") or {}).get("state_fit_heave_rms_mean"),
                    "fit_status": (item.get("frequency_rms") or {}).get("fit_status"),
                    "elapsed_seconds": item["elapsed_seconds"],
                }
            )
    return path


def plot_error_matrix(path: Path, cases: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item["case_id"] for item in cases]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8.4, 0.6 * len(labels)), 4.8))
    ax.bar(
        x - width,
        [item["comparison"]["state_vs_direct_master_displacement_l2_relative_error"] for item in cases],
        width,
        label="master displacement",
        color="#111111",
    )
    ax.bar(
        x,
        [item["comparison"]["state_vs_direct_memory_force_l2_relative_error"] for item in cases],
        width,
        label="memory force",
        color="#1f77b4",
    )
    ax.bar(
        x + width,
        [item["comparison"]["state_vs_direct_centerline_heave_rms_relative_error"] for item in cases],
        width,
        label="heave RMS",
        color="#d62728",
    )
    ax.set_yscale("log")
    ax.set_ylabel("state/direct relative error")
    ax.set_title("ERA state-space vs direct Cummins across sea states")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_heave_rms_summary(path: Path, cases: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item["case_id"] for item in cases]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.4, 0.6 * len(labels)), 4.8))
    ax.plot(
        x,
        [item["direct"]["centerline_heave_rms_mean"] for item in cases],
        color="#111111",
        linewidth=1.5,
        marker="o",
        label="direct Cummins",
    )
    ax.plot(
        x,
        [item["state_space"]["centerline_heave_rms_mean"] for item in cases],
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        marker="s",
        label="state-space",
    )
    if cases and "frequency_rms" in cases[0]:
        ax.plot(
            x,
            [item["frequency_rms"]["frequency_centerline_heave_rms_mean"] for item in cases],
            color="#1f77b4",
            linestyle=":",
            linewidth=1.4,
            marker="^",
            label="frequency spectrum",
        )
    ax.set_ylabel("mean centerline heave RMS")
    ax.set_title("Heave RMS output comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_frequency_time_rms_errors(path: Path, cases: list[dict[str, object]]) -> Path | None:
    if not cases or "frequency_rms" not in cases[0]:
        return None
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item["case_id"] for item in cases]
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(max(8.4, 0.6 * len(labels)), 4.8))
    ax.bar(
        x - width / 2,
        [
            item["frequency_rms"].get("frequency_vs_direct_fit_rms_l2_relative_error")
            or item["frequency_rms"]["frequency_vs_direct_post_discard_rms_l2_relative_error"]
            for item in cases
        ],
        width,
        color="#111111",
        label="direct fit/post RMS",
    )
    ax.bar(
        x + width / 2,
        [
            item["frequency_rms"].get("frequency_vs_state_fit_rms_l2_relative_error")
            or item["frequency_rms"]["frequency_vs_state_post_discard_rms_l2_relative_error"]
            for item in cases
        ],
        width,
        color="#d62728",
        label="state-space fit/post RMS",
    )
    ax.set_ylabel("relative error vs frequency-domain RMS")
    ax.set_title("Frequency-domain RMS vs time-domain RMS")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_long_window_metrics(path: Path, long_case: dict[str, object]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    windows = np.arange(1, len(long_case["centerline_heave_window_rms_mean"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    if long_case["wave_window_hs"] is not None:
        axes[0].plot(windows, long_case["wave_window_hs"], color="#2ca02c", marker="o", linewidth=1.4)
        axes[0].axhline(long_case["significant_wave_height"], color="#555555", linestyle=":", linewidth=1.0)
        axes[0].set_ylabel("window Hs")
    else:
        axes[0].axis("off")
    axes[1].plot(
        windows,
        long_case["centerline_heave_window_rms_mean"],
        color="#111111",
        marker="o",
        linewidth=1.4,
        label="mean",
    )
    axes[1].plot(
        windows,
        long_case["centerline_heave_window_rms_max"],
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=1.2,
        label="max",
    )
    axes[1].set_xlabel("time window")
    axes[1].set_ylabel("centerline heave RMS")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.suptitle("Long lightweight state-space screening")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_report(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = metrics["short_validation_cases"]
    max_heave_rms_error = max(
        item["comparison"]["state_vs_direct_centerline_heave_rms_relative_error"]
        for item in cases
    )
    max_master_error = max(
        item["comparison"]["state_vs_direct_master_displacement_l2_relative_error"]
        for item in cases
    )
    frequency_cases = [item for item in cases if "frequency_rms" in item]
    lines = [
        "# WEC-Sim-like multi-sea-state validation",
        "",
        f"- generated_at: `{metrics['generated_at']}`",
        f"- status: `{metrics['status']}`",
        f"- adapter_layer_only: `{metrics['adapter_layer_only']}`",
        f"- rodm_frequency_core_modified: `{metrics['rodm_frequency_core_modified']}`",
        f"- mooring_module_modified: `{metrics['mooring_module_modified']}`",
        f"- short_cases: `{len(cases)}`",
        f"- max state/direct master displacement error: `{max_master_error:.6g}`",
        f"- max state/direct heave RMS error: `{max_heave_rms_error:.6g}`",
        "",
        "## Outputs",
        "",
        f"- summary_csv: `{metrics['summary_csv']}`",
    ]
    if frequency_cases:
        max_frequency_state = max(
            item["frequency_rms"]["frequency_vs_state_post_discard_rms_l2_relative_error"]
            for item in frequency_cases
        )
        max_frequency_direct = max(
            item["frequency_rms"]["frequency_vs_direct_post_discard_rms_l2_relative_error"]
            for item in frequency_cases
        )
        lines.extend(
            [
                f"- max frequency/direct time RMS error: `{max_frequency_direct:.6g}`",
                f"- max frequency/state time RMS error: `{max_frequency_state:.6g}`",
            ]
        )
        fit_cases = [
            item for item in frequency_cases
            if item["frequency_rms"].get("fit_status") == "completed"
        ]
        if fit_cases:
            max_fit_direct = max(
                item["frequency_rms"]["frequency_vs_direct_fit_rms_l2_relative_error"]
                for item in fit_cases
            )
            max_fit_state = max(
                item["frequency_rms"]["frequency_vs_state_fit_rms_l2_relative_error"]
                for item in fit_cases
            )
            lines.extend(
                [
                    f"- max frequency/direct fitted RMS error: `{max_fit_direct:.6g}`",
                    f"- max frequency/state fitted RMS error: `{max_fit_state:.6g}`",
                    f"- fitted RMS cases: `{len(fit_cases)}`",
                ]
            )
    if "long_lightweight_case" in metrics:
        long_case = metrics["long_lightweight_case"]
        lines.extend(
            [
                "",
                "## Long Lightweight Case",
                "",
                f"- case_id: `{long_case['case_id']}`",
                f"- duration_s: `{long_case['duration_s']}`",
                f"- time_samples: `{long_case['time_samples']}`",
                f"- output mode: `{long_case['lightweight_output']}`",
            ]
        )
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{figure}`" for figure in metrics["figures"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def json_ready_case(item: dict[str, object]) -> dict[str, object]:
    ready = dict(item)
    ready["direct_centerline_heave_rms"] = np.asarray(item["direct_centerline_heave_rms"]).tolist()
    ready["state_centerline_heave_rms"] = np.asarray(item["state_centerline_heave_rms"]).tolist()
    if "frequency_rms" in ready:
        frequency_rms = dict(ready["frequency_rms"])
        for key in (
            "state_vs_frequency_centerline_heave_rms",
            "direct_vs_frequency_centerline_heave_rms",
            "frequency_centerline_heave_rms",
            "direct_fit_centerline_heave_rms",
            "state_fit_centerline_heave_rms",
        ):
            if key in frequency_rms:
                frequency_rms[key] = np.asarray(frequency_rms[key]).tolist()
        ready["frequency_rms"] = frequency_rms
    return ready


def json_ready_long(item: dict[str, object]) -> dict[str, object]:
    ready = dict(item)
    ready["centerline_heave_window_rms_mean"] = np.asarray(item["centerline_heave_window_rms_mean"]).tolist()
    ready["centerline_heave_window_rms_max"] = np.asarray(item["centerline_heave_window_rms_max"]).tolist()
    if item["wave_window_hs"] is not None:
        ready["wave_window_hs"] = np.asarray(item["wave_window_hs"]).tolist()
    return ready


def main() -> int:
    args = parse_args()
    start = timer.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    hs_values = parse_float_list(args.hs_values)
    target_omega_values = parse_float_list(args.target_omega_values)
    seeds = parse_int_list(args.seeds)
    frequency_omega = None
    frequency_heave_rao = None
    frequency_elapsed = None
    if not args.skip_frequency_rms:
        frequency_start = timer.perf_counter()
        frequency_omega, frequency_heave_rao = solve_frequency_centerline_rao(args)
        frequency_elapsed = float(timer.perf_counter() - frequency_start)
        print(
            "frequency-domain RAO precompute completed: "
            f"{frequency_omega.size} omega points in {frequency_elapsed:.3f}s"
        )

    cases: list[dict[str, object]] = []
    for hs in hs_values:
        for omega in target_omega_values:
            for seed in seeds:
                item = run_short_validation_case(
                    args,
                    hs=hs,
                    target_omega=omega,
                    seed=seed,
                    output_root=output_root / "cases" / case_id(hs, omega, seed),
                    frequency_omega=frequency_omega,
                    frequency_heave_rao=frequency_heave_rao,
                )
                cases.append(item)
                print(
                    f"{item['case_id']}: master={item['comparison']['state_vs_direct_master_displacement_l2_relative_error']:.6g}, "
                    f"memory={item['comparison']['state_vs_direct_memory_force_l2_relative_error']:.6g}, "
                    f"heave_rms={item['comparison']['state_vs_direct_centerline_heave_rms_relative_error']:.6g}"
                )

    figures = [
        plot_error_matrix(output_root / "figures" / "state_space_direct_error_matrix.png", cases),
        plot_heave_rms_summary(output_root / "figures" / "centerline_heave_rms_summary.png", cases),
    ]
    frequency_figure = plot_frequency_time_rms_errors(
        output_root / "figures" / "frequency_time_rms_errors.png",
        cases,
    )
    if frequency_figure is not None:
        figures.append(frequency_figure)
    long_case = None
    if not args.skip_long_run:
        long_case = run_long_lightweight_case(args, output_root / "long_lightweight")
        figures.append(
            plot_long_window_metrics(
                output_root / "figures" / "long_lightweight_window_metrics.png",
                long_case,
            )
        )

    summary_csv = write_summary_csv(output_root / "multi_sea_state_summary.csv", cases)
    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "mooring_module_modified": False,
        "hydrodynamic_file": args.hydro_file,
        "short_validation_settings": {
            "hs_values": hs_values,
            "target_omega_values": target_omega_values,
            "seeds": seeds,
            "cycles": args.cycles,
            "steps_per_cycle": args.steps_per_cycle,
            "memory_cycles": args.memory_cycles,
            "state_order": args.state_order,
            "era_block_rows": args.era_block_rows,
            "era_block_cols": args.era_block_cols,
            "integrator": args.integrator,
            "radiation_convolution_rule": args.radiation_convolution_rule,
            "radiation_residual_model": args.radiation_residual_model,
            "mooring_corner_horizontal_stiffness": args.mooring_corner_horizontal_stiffness,
            "frequency_rms_enabled": not args.skip_frequency_rms,
            "frequency_rms_discard_cycles": args.frequency_rms_discard_cycles,
            "fit_min_samples_per_parameter": args.fit_min_samples_per_parameter,
        },
        "short_validation_cases": [json_ready_case(item) for item in cases],
        "summary_csv": summary_csv,
        "figures": figures,
    }
    if frequency_elapsed is not None:
        metrics["frequency_rao_precompute"] = {
            "elapsed_seconds": frequency_elapsed,
            "omega_count": int(frequency_omega.size),
        }
    if long_case is not None:
        metrics["long_lightweight_case"] = json_ready_long(long_case)
    metrics["elapsed_seconds"] = float(timer.perf_counter() - start)
    metrics_path = write_metrics_json(output_root / "multi_sea_state_metrics.json", metrics)
    report_path = write_report(output_root / "report.md", metrics)

    max_master = max(
        item["comparison"]["state_vs_direct_master_displacement_l2_relative_error"]
        for item in cases
    )
    max_heave = max(
        item["comparison"]["state_vs_direct_centerline_heave_rms_relative_error"]
        for item in cases
    )
    print("Multi-sea-state WEC-Sim-like validation completed.")
    print(f"short_cases: {len(cases)}")
    print(f"max_master_error: {max_master:.6g}")
    print(f"max_heave_rms_error: {max_heave:.6g}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
