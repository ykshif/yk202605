"""Sweep Cummins time-step and memory-duration settings for spectrum validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_spectrum_frequency_time_response import solve_centerline_frequency_rao  # noqa: E402
from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from validate_spectrum_seed_sweep import load_array, validate_seed_case  # noqa: E402
from run_time_domain_reference_case_300 import default_dm_fem_root  # noqa: E402


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_spectrum_dense_88_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "num_sens_dense88"
DEFAULT_TARGET_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--memory-cycles", type=str, default="2,4,6")
    parser.add_argument("--steps-per-peak-cycle", type=str, default="30,40,60")
    parser.add_argument("--spectrum-seed", type=int, default=20260522)
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--discard-peak-cycles", type=float, default=5.0)
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
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one float value is required")
    if any(item <= 0.0 for item in values):
        raise ValueError("all float values must be positive")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    if any(item < 2 for item in values):
        raise ValueError("integer values must be at least 2")
    return values


def token(value: float | int) -> str:
    text = f"{float(value):g}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "m")


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_sensitivity_case(
    *,
    args: argparse.Namespace,
    data_root: Path,
    hydro_path: Path,
    memory_cycles: float,
    steps_per_peak_cycle: int,
    case_root: Path,
) -> None:
    if args.skip_existing and (case_root / "metrics.json").exists():
        return
    runner = REPO_ROOT / "scripts" / "run_time_domain_excitation_case.py"
    run_command(
        [
            sys.executable,
            str(runner),
            "--data-root",
            str(data_root),
            "--hydro-file",
            str(hydro_path),
            "--output-root",
            str(case_root),
            "--excitation-model",
            "wave_spectrum",
            "--significant-wave-height",
            str(args.significant_wave_height),
            "--spectrum-type",
            args.spectrum_type,
            "--peak-enhancement-factor",
            str(args.peak_enhancement_factor),
            "--spectrum-seed",
            str(args.spectrum_seed),
            "--target-omega",
            str(args.target_omega),
            "--peak-cycles",
            str(args.peak_cycles),
            "--steps-per-peak-cycle",
            str(steps_per_peak_cycle),
            "--memory-cycles",
            str(memory_cycles),
            "--hydro-node-order",
            args.hydro_node_order,
            "--radiation-passivity-correction",
            args.radiation_passivity_correction,
            "--radiation-convolution-rule",
            args.radiation_convolution_rule,
            "--radiation-residual-model",
            args.radiation_residual_model,
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = [
        "memory_cycles",
        "steps_per_peak_cycle",
        "time_step_s",
        "memory_duration_s",
        "time_samples",
        "frequency_vs_time_fit_rms_l2_relative_error",
        "frequency_vs_time_series_rms_l2_relative_error",
        "motion_spectrum_density_l2_relative_error",
        "centerline_heave_rms_max",
        "centerline_heave_abs_max",
        "elapsed_seconds",
        "case_root",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    return path


def metric_grid(
    rows: list[dict[str, object]],
    memory_values: list[float],
    step_values: list[int],
    metric_name: str,
) -> np.ndarray:
    grid = np.full((len(memory_values), len(step_values)), np.nan, dtype=float)
    memory_lookup = {value: index for index, value in enumerate(memory_values)}
    step_lookup = {value: index for index, value in enumerate(step_values)}
    for row in rows:
        i = memory_lookup[float(row["memory_cycles"])]
        j = step_lookup[int(row["steps_per_peak_cycle"])]
        grid[i, j] = float(row[metric_name])
    return grid


def plot_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    memory_values: list[float],
    step_values: list[int],
    metric_name: str,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    grid = metric_grid(rows, memory_values, step_values, metric_name)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(step_values)))
    ax.set_xticklabels([str(value) for value in step_values])
    ax.set_yticks(np.arange(len(memory_values)))
    ax.set_yticklabels([f"{value:g}" for value in memory_values])
    ax.set_xlabel("Steps per peak cycle")
    ax.set_ylabel("Memory cycles")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3g}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label=metric_name)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_metric_lines(
    path: Path,
    rows: list[dict[str, object]],
    memory_values: list[float],
    step_values: list[int],
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if len(step_values) == 1:
        subset = sorted(rows, key=lambda item: float(item["memory_cycles"]))
        ax.plot(
            [float(row["memory_cycles"]) for row in subset],
            [float(row["frequency_vs_time_fit_rms_l2_relative_error"]) for row in subset],
            marker="o",
            linewidth=1.5,
            label=f"{step_values[0]} steps/cycle",
        )
        ax.set_xlabel("Memory cycles")
    else:
        for memory in memory_values:
            subset = sorted(
                [row for row in rows if float(row["memory_cycles"]) == memory],
                key=lambda item: int(item["steps_per_peak_cycle"]),
            )
            ax.plot(
                [int(row["steps_per_peak_cycle"]) for row in subset],
                [float(row["frequency_vs_time_fit_rms_l2_relative_error"]) for row in subset],
                marker="o",
                linewidth=1.5,
                label=f"memory {memory:g} cycles",
            )
        ax.set_xlabel("Steps per peak cycle")
    ax.set_ylabel("Frequency/time fit RMS relative error")
    ax.set_title("Cummins numerical sensitivity")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def summarize(rows: list[dict[str, object]], metric_name: str) -> dict[str, float]:
    values = np.array([float(row[metric_name]) for row in rows], dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def best_row(rows: list[dict[str, object]], metric_name: str) -> dict[str, object]:
    return min(rows, key=lambda row: float(row[metric_name]))


def main() -> int:
    args = parse_args()
    start = timer.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    memory_values = parse_float_list(args.memory_cycles)
    step_values = parse_int_list(args.steps_per_peak_cycle)

    case_specs = [
        (
            memory_cycles,
            steps_per_peak_cycle,
            output_root
            / "cases"
            / f"mem_{token(memory_cycles)}_steps_{token(steps_per_peak_cycle)}",
        )
        for memory_cycles in memory_values
        for steps_per_peak_cycle in step_values
    ]
    for memory_cycles, steps_per_peak_cycle, case_root in case_specs:
        run_sensitivity_case(
            args=args,
            data_root=data_root,
            hydro_path=hydro_path,
            memory_cycles=memory_cycles,
            steps_per_peak_cycle=steps_per_peak_cycle,
            case_root=case_root,
        )

    first_omega = load_array(case_specs[0][2], "wave_component_omega.npy")
    frequency_heave_rao = solve_centerline_frequency_rao(
        hydro_path=hydro_path,
        data_root=data_root,
        component_omega=first_omega,
        reversed_hydro=args.hydro_node_order == "reversed",
    )

    rows: list[dict[str, object]] = []
    for memory_cycles, steps_per_peak_cycle, case_root in case_specs:
        validation = validate_seed_case(
            case_root=case_root,
            seed=args.spectrum_seed,
            frequency_heave_rao=frequency_heave_rao,
            discard_peak_cycles=args.discard_peak_cycles,
        )
        case_metrics = json.loads((case_root / "metrics.json").read_text(encoding="utf-8"))
        row = {
            **validation,
            "memory_cycles": float(memory_cycles),
            "steps_per_peak_cycle": int(steps_per_peak_cycle),
            "time_step_s": float(case_metrics["time_step_s"]),
            "memory_duration_s": float(memory_cycles) * float(case_metrics["peak_period_s"]),
            "centerline_heave_abs_max": float(case_metrics["centerline_heave_abs_max"]),
            "elapsed_seconds": float(case_metrics["elapsed_seconds"]),
        }
        rows.append(row)

    csv_path = write_csv(output_root / "cummins_numerical_sensitivity.csv", rows)
    figures = [
        plot_heatmap(
            output_root / "figures" / "fit_rms_error_heatmap.png",
            rows,
            memory_values,
            step_values,
            "frequency_vs_time_fit_rms_l2_relative_error",
            "Frequency/time harmonic-fit RMS error",
        ),
        plot_heatmap(
            output_root / "figures" / "motion_spectrum_error_heatmap.png",
            rows,
            memory_values,
            step_values,
            "motion_spectrum_density_l2_relative_error",
            "Motion-spectrum density error",
        ),
        plot_heatmap(
            output_root / "figures" / "time_series_rms_error_heatmap.png",
            rows,
            memory_values,
            step_values,
            "frequency_vs_time_series_rms_l2_relative_error",
            "Direct time-series RMS error",
        ),
        plot_metric_lines(
            output_root / "figures" / "fit_rms_error_lines.png",
            rows,
            memory_values,
            step_values,
        ),
    ]
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "hydrodynamic_dataset": hydro_path,
        "target_omega_rad_s": args.target_omega,
        "spectrum_seed": args.spectrum_seed,
        "memory_cycles": memory_values,
        "steps_per_peak_cycle": step_values,
        "radiation_residual_model": args.radiation_residual_model,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "radiation_passivity_correction": args.radiation_passivity_correction,
        "component_count": int(first_omega.size),
        "case_count": len(rows),
        "elapsed_seconds": timer.perf_counter() - start,
        "summary": {
            "frequency_vs_time_fit_rms_l2_relative_error": summarize(
                rows,
                "frequency_vs_time_fit_rms_l2_relative_error",
            ),
            "motion_spectrum_density_l2_relative_error": summarize(
                rows,
                "motion_spectrum_density_l2_relative_error",
            ),
            "frequency_vs_time_series_rms_l2_relative_error": summarize(
                rows,
                "frequency_vs_time_series_rms_l2_relative_error",
            ),
        },
        "best": {
            "frequency_vs_time_fit_rms_l2_relative_error": best_row(
                rows,
                "frequency_vs_time_fit_rms_l2_relative_error",
            ),
            "motion_spectrum_density_l2_relative_error": best_row(
                rows,
                "motion_spectrum_density_l2_relative_error",
            ),
        },
        "rows": rows,
        "csv": csv_path,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "cummins_numerical_sensitivity_metrics.json", metrics)

    print("Cummins numerical sensitivity validation completed.")
    print(f"case_count: {len(rows)}")
    print(
        "fit_rms_error min/max: "
        f"{metrics['summary']['frequency_vs_time_fit_rms_l2_relative_error']['min']:.6g} / "
        f"{metrics['summary']['frequency_vs_time_fit_rms_l2_relative_error']['max']:.6g}"
    )
    print(
        "motion_spectrum_error min/max: "
        f"{metrics['summary']['motion_spectrum_density_l2_relative_error']['min']:.6g} / "
        f"{metrics['summary']['motion_spectrum_density_l2_relative_error']['max']:.6g}"
    )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
