"""Sweep ERA state-space radiation order for the Cummins adapter."""

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

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    direct_convolution_memory_force,
    radiation_coefficients_from_discrete_irf,
    spectral_component_widths,
    wave_spectrum_density,
)
from offshore_energy_sim.time_domain.rodm_hydrodynamics import (  # noqa: E402
    _reduced_matrix_series,
    prepare_rodm_time_domain_hydrodynamic_terms,
)
from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    evaluate_era_radiation_kernel,
    fit_era_state_space_radiation,
    simulate_era_memory_force,
)
from run_time_domain_reference_case_300 import build_default_case, default_dm_fem_root  # noqa: E402
from validate_state_space_radiation import (  # noqa: E402
    DEFAULT_HYDRO,
    DEFAULT_TARGET_OMEGA,
    DEFAULT_VELOCITY_CASE,
    dominant_diagonal_index,
    load_array,
    relative_l2,
    weighted_relative_series_error,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "state_space_order_sweep_dense88_era"


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--state-orders", type=parse_int_list, default=parse_int_list("40,80,120,160,200,240"))
    parser.add_argument("--era-block-rows", type=int, default=55)
    parser.add_argument("--era-block-cols", type=int, default=55)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=60)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
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
    parser.add_argument("--velocity-case-root", type=Path, default=DEFAULT_VELOCITY_CASE)
    parser.add_argument("--memory-force-samples", type=int, default=1200)
    parser.add_argument("--target-kernel-error", type=float, default=1.0e-2)
    parser.add_argument("--target-memory-force-error", type=float, default=2.0e-2)
    return parser.parse_args()


def direct_convolution_memory_history(
    velocity: np.ndarray,
    kernel: np.ndarray,
    time: np.ndarray,
    *,
    convolution_rule: str,
) -> np.ndarray:
    force = np.zeros_like(velocity)
    dt = float(np.diff(time)[0])
    for index in range(1, time.size):
        force[index] = direct_convolution_memory_force(
            velocity,
            kernel,
            dt,
            index,
            convolution_rule=convolution_rule,
        )
    return force


def plot_order_sweep(path: Path, rows: list[dict[str, float]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    orders = np.asarray([row["model_order"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(orders, [row["kernel_fit_l2_relative_error"] for row in rows], marker="o", label="kernel fit")
    ax.plot(orders, [row["state_vs_direct_added_weighted_error"] for row in rows], marker="s", label="A vs direct")
    ax.plot(orders, [row["state_vs_direct_damping_weighted_error"] for row in rows], marker="^", label="B vs direct")
    ax.plot(orders, [row["state_vs_direct_memory_force_l2_relative_error"] for row in rows], marker="d", label="memory force")
    ax.set_yscale("log")
    ax.set_xlabel("ERA retained order")
    ax.set_ylabel("Relative error")
    ax.set_title("ERA state-space radiation order sweep")
    ax.grid(True, color="#dddddd", linewidth=0.7, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_memory_rms(path: Path, rows: list[dict[str, float]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    orders = np.asarray([row["model_order"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(orders, [row["direct_memory_force_rms"] for row in rows], color="#111111", marker="o", label="direct")
    ax.plot(orders, [row["state_memory_force_rms"] for row in rows], color="#d62728", marker="s", label="state-space")
    ax.set_xlabel("ERA retained order")
    ax.set_ylabel("Radiation-memory force RMS")
    ax.set_title("Memory-force RMS by ERA order")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_order",
        "model_order",
        "spectral_radius",
        "kernel_fit_l2_relative_error",
        "state_vs_direct_added_weighted_error",
        "state_vs_direct_damping_weighted_error",
        "selected_state_vs_direct_added_error",
        "selected_state_vs_direct_damping_error",
        "state_vs_direct_memory_force_l2_relative_error",
        "direct_memory_force_rms",
        "state_memory_force_rms",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def select_candidate(
    rows: list[dict[str, float]],
    *,
    target_kernel_error: float,
    target_memory_force_error: float,
) -> dict[str, float] | None:
    feasible = [
        row
        for row in rows
        if row["kernel_fit_l2_relative_error"] <= target_kernel_error
        and row["state_vs_direct_memory_force_l2_relative_error"] <= target_memory_force_error
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda row: row["model_order"])


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
            radiation_model="direct_convolution",
            memory_duration=args.memory_cycles * peak_period,
            radiation_passivity_correction=args.radiation_passivity_correction,
            radiation_convolution_rule=args.radiation_convolution_rule,
            radiation_residual_model=args.radiation_residual_model,
        )
        terms = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)
        order = np.argsort(omega_grid)
        omega = omega_grid[order]
        original_added = _reduced_matrix_series(dataset["added_mass"].values, case)[order]
        original_damping = _reduced_matrix_series(dataset["radiation_damping"].values, case)[order]
    finally:
        dataset.close()

    if terms.radiation_irf is None or terms.radiation_irf_time is None or terms.added_mass_infinite is None:
        raise RuntimeError("direct-convolution preprocessing did not produce radiation IRF terms")

    direct_added, direct_damping = radiation_coefficients_from_discrete_irf(
        omega,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
        convolution_rule=args.radiation_convolution_rule,
    )
    residual_added = np.zeros_like(direct_added[0]) if terms.residual_added_mass is None else terms.residual_added_mass
    residual_damping = (
        np.zeros_like(direct_damping[0])
        if terms.residual_radiation_damping is None
        else terms.residual_radiation_damping
    )
    direct_added_corrected = direct_added + residual_added[np.newaxis, :, :]
    direct_damping_corrected = direct_damping + residual_damping[np.newaxis, :, :]

    density = wave_spectrum_density(
        omega,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=peak_period,
        gamma=args.peak_enhancement_factor,
    )
    weights = density * spectral_component_widths(omega)
    weights = weights / max(float(np.sum(weights)), 1.0e-30)
    selected = int(np.argmin(np.abs(omega - selected_omega)))
    dof = dominant_diagonal_index(original_damping)

    velocity_case_root = args.velocity_case_root
    if velocity_case_root is None or not velocity_case_root.exists():
        raise FileNotFoundError(f"velocity case root not found: {velocity_case_root}")
    velocity = load_array(velocity_case_root, "master_velocity_time.npy")
    velocity_time = load_array(velocity_case_root, "time.npy")
    sample_count = min(int(args.memory_force_samples), velocity_time.size)
    velocity = velocity[:sample_count]
    velocity_time = velocity_time[:sample_count]
    direct_memory = direct_convolution_memory_history(
        velocity,
        terms.radiation_irf,
        velocity_time,
        convolution_rule=args.radiation_convolution_rule,
    )
    direct_memory_rms = float(np.sqrt(np.mean(np.linalg.norm(direct_memory, axis=1) ** 2)))

    rows: list[dict[str, float]] = []
    for requested_order in args.state_orders:
        case_start = timer.perf_counter()
        model = fit_era_state_space_radiation(
            terms.radiation_irf_time,
            terms.radiation_irf,
            order=requested_order,
            block_rows=args.era_block_rows,
            block_cols=args.era_block_cols,
        )
        state_kernel = evaluate_era_radiation_kernel(model, terms.radiation_irf_time.size)
        state_added, state_damping = radiation_coefficients_from_discrete_irf(
            omega,
            state_kernel,
            terms.radiation_irf_time,
            added_mass_infinite=terms.added_mass_infinite,
            convolution_rule=args.radiation_convolution_rule,
        )
        state_memory = simulate_era_memory_force(velocity, velocity_time, model)
        state_added_corrected = state_added + residual_added[np.newaxis, :, :]
        state_damping_corrected = state_damping + residual_damping[np.newaxis, :, :]
        row = {
            "requested_order": int(requested_order),
            "model_order": int(model.order),
            "era_block_rows": int(args.era_block_rows),
            "era_block_cols": int(args.era_block_cols),
            "spectral_radius": float(model.spectral_radius),
            "kernel_fit_l2_relative_error": float(model.fit_l2_relative_error),
            "state_added_corrected_weighted_error": weighted_relative_series_error(
                state_added_corrected,
                original_added,
                weights,
            ),
            "state_damping_corrected_weighted_error": weighted_relative_series_error(
                state_damping_corrected,
                original_damping,
                weights,
            ),
            "state_vs_direct_added_weighted_error": weighted_relative_series_error(
                state_added,
                direct_added,
                weights,
            ),
            "state_vs_direct_damping_weighted_error": weighted_relative_series_error(
                state_damping,
                direct_damping,
                weights,
            ),
            "selected_state_vs_direct_added_error": relative_l2(
                state_added[selected],
                direct_added[selected],
            ),
            "selected_state_vs_direct_damping_error": relative_l2(
                state_damping[selected],
                direct_damping[selected],
            ),
            "state_vs_direct_memory_force_l2_relative_error": relative_l2(
                state_memory,
                direct_memory,
            ),
            "direct_memory_force_rms": direct_memory_rms,
            "state_memory_force_rms": float(np.sqrt(np.mean(np.linalg.norm(state_memory, axis=1) ** 2))),
            "elapsed_seconds": float(timer.perf_counter() - case_start),
        }
        rows.append(row)
        print(
            "order "
            f"{row['model_order']}: kernel={row['kernel_fit_l2_relative_error']:.6g}, "
            f"A={row['state_vs_direct_added_weighted_error']:.6g}, "
            f"B={row['state_vs_direct_damping_weighted_error']:.6g}, "
            f"memory={row['state_vs_direct_memory_force_l2_relative_error']:.6g}"
        )

    candidate = select_candidate(
        rows,
        target_kernel_error=args.target_kernel_error,
        target_memory_force_error=args.target_memory_force_error,
    )
    figures = [
        plot_order_sweep(output_root / "figures" / "state_space_order_sweep_errors.png", rows),
        plot_memory_rms(output_root / "figures" / "state_space_order_sweep_memory_rms.png", rows),
    ]
    csv_path = write_csv(output_root / "state_space_order_sweep.csv", rows)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "hydrodynamic_dataset": hydro_path,
        "selected_omega_rad_s": selected_omega,
        "memory_cycles": args.memory_cycles,
        "steps_per_peak_cycle": args.steps_per_peak_cycle,
        "memory_time_count": int(terms.radiation_irf_time.size),
        "memory_duration_s": float(terms.radiation_irf_time[-1]),
        "component_count": int(omega.size),
        "dominant_diagonal_dof": int(dof),
        "era_block_rows": int(args.era_block_rows),
        "era_block_cols": int(args.era_block_cols),
        "target_kernel_error": float(args.target_kernel_error),
        "target_memory_force_error": float(args.target_memory_force_error),
        "direct_reference": {
            "direct_added_corrected_weighted_error": weighted_relative_series_error(
                direct_added_corrected,
                original_added,
                weights,
            ),
            "direct_damping_corrected_weighted_error": weighted_relative_series_error(
                direct_damping_corrected,
                original_damping,
                weights,
            ),
            "direct_memory_force_rms": direct_memory_rms,
            "velocity_case_root": velocity_case_root,
            "sample_count": sample_count,
        },
        "recommended_candidate": candidate,
        "orders": rows,
        "csv": csv_path,
        "figures": figures,
        "elapsed_seconds": float(timer.perf_counter() - start),
    }
    metrics_path = write_metrics_json(output_root / "state_space_order_sweep_metrics.json", metrics)
    print("State-space ERA order sweep completed.")
    if candidate is None:
        print("recommended_candidate: none")
    else:
        print(
            "recommended_candidate: "
            f"order {candidate['model_order']}, "
            f"kernel={candidate['kernel_fit_l2_relative_error']:.6g}, "
            f"memory={candidate['state_vs_direct_memory_force_l2_relative_error']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
