"""Validate spectrum-driven Cummins response over multiple random wave seeds."""

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

from compare_spectrum_frequency_time_response import (  # noqa: E402
    motion_spectral_density,
    solve_centerline_frequency_rao,
)
from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_multi_harmonic_amplitudes,
    relative_l2_error,
    zero_mean_rms,
)
from run_time_domain_reference_case_300 import default_dm_fem_root  # noqa: E402


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_spectrum_dense_88_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "seed_sweep_dense88"
DEFAULT_TARGET_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated random phase seeds.")
    parser.add_argument("--seed-start", type=int, default=20260522)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=40)
    parser.add_argument("--discard-peak-cycles", type=float, default=5.0)
    parser.add_argument("--memory-cycles", type=float, default=4.0)
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


def parse_seeds(value: str | None, *, start: int, count: int) -> list[int]:
    if value is None:
        if count < 1:
            raise ValueError("--seed-count must be positive")
        return [start + offset for offset in range(count)]
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds did not contain any integers")
    return seeds


def load_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_seed_case(args: argparse.Namespace, hydro_path: Path, data_root: Path, seed: int, case_root: Path) -> None:
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
            str(seed),
            "--target-omega",
            str(args.target_omega),
            "--peak-cycles",
            str(args.peak_cycles),
            "--steps-per-peak-cycle",
            str(args.steps_per_peak_cycle),
            "--memory-cycles",
            str(args.memory_cycles),
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


def validate_seed_case(
    *,
    case_root: Path,
    seed: int,
    frequency_heave_rao: np.ndarray,
    discard_peak_cycles: float,
) -> dict[str, object]:
    metrics = load_json(case_root / "metrics.json")
    time = load_array(case_root, "time.npy")
    omega = load_array(case_root, "wave_component_omega.npy")
    wave_amplitude = load_array(case_root, "wave_component_amplitude.npy")
    heave = load_array(case_root, "centerline_heave_time.npy")

    discard_seconds = discard_peak_cycles * float(metrics["peak_period_s"])
    mask = time >= time[0] + discard_seconds
    if np.count_nonzero(mask) <= 2 * omega.size + 1:
        raise ValueError(f"seed {seed} does not have enough samples after discard")
    fit_start = float(time[mask][0])

    time_heave_components = fit_multi_harmonic_amplitudes(
        heave,
        time,
        omega,
        start_time=fit_start,
    )
    frequency_components = frequency_heave_rao * wave_amplitude[:, np.newaxis]
    frequency_density = motion_spectral_density(frequency_components, omega)
    time_density = motion_spectral_density(time_heave_components, omega)
    frequency_rms = np.sqrt(0.5 * np.sum(np.abs(frequency_components) ** 2, axis=0))
    time_fit_rms = np.sqrt(0.5 * np.sum(np.abs(time_heave_components) ** 2, axis=0))
    time_series_rms = zero_mean_rms(heave[mask], axis=0)

    return {
        "seed": seed,
        "case_root": case_root,
        "time_samples": int(time.size),
        "component_count": int(omega.size),
        "discard_seconds": discard_seconds,
        "fit_start_time_s": fit_start,
        "centerline_heave_rms_max": float(metrics["centerline_heave_rms_max"]),
        "frequency_vs_time_fit_rms_l2_relative_error": relative_l2_error(time_fit_rms, frequency_rms),
        "frequency_vs_time_series_rms_l2_relative_error": relative_l2_error(time_series_rms, frequency_rms),
        "motion_spectrum_density_l2_relative_error": relative_l2_error(time_density, frequency_density),
        "frequency_rms_max": float(np.max(frequency_rms)),
        "time_fit_rms_max": float(np.max(time_fit_rms)),
        "time_series_rms_max": float(np.max(time_series_rms)),
        "time_series_minus_frequency_rms_mean": float(np.mean(time_series_rms - frequency_rms)),
        "time_fit_minus_frequency_rms_mean": float(np.mean(time_fit_rms - frequency_rms)),
    }


def write_seed_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "component_count",
        "time_samples",
        "frequency_vs_time_fit_rms_l2_relative_error",
        "frequency_vs_time_series_rms_l2_relative_error",
        "motion_spectrum_density_l2_relative_error",
        "frequency_rms_max",
        "time_fit_rms_max",
        "time_series_rms_max",
        "time_fit_minus_frequency_rms_mean",
        "time_series_minus_frequency_rms_mean",
        "case_root",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    return path


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def plot_seed_errors(path: Path, rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [str(row["seed"]) for row in rows]
    fit = [float(row["frequency_vs_time_fit_rms_l2_relative_error"]) for row in rows]
    series = [float(row["frequency_vs_time_series_rms_l2_relative_error"]) for row in rows]
    spectrum = [float(row["motion_spectrum_density_l2_relative_error"]) for row in rows]

    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - width, fit, width=width, label="harmonic-fit RMS")
    ax.bar(x, series, width=width, label="time-series RMS")
    ax.bar(x + width, spectrum, width=width, label="motion spectrum")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=35, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Spectrum time-domain validation over random seeds")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    seeds = parse_seeds(args.seeds, start=args.seed_start, count=args.seed_count)

    start = timer.perf_counter()
    case_roots = [output_root / "cases" / f"seed_{seed}" for seed in seeds]
    for seed, case_root in zip(seeds, case_roots):
        run_seed_case(args, hydro_path, data_root, seed, case_root)

    first_omega = load_array(case_roots[0], "wave_component_omega.npy")
    frequency_heave_rao = solve_centerline_frequency_rao(
        hydro_path=hydro_path,
        data_root=data_root,
        component_omega=first_omega,
        reversed_hydro=args.hydro_node_order == "reversed",
    )
    rows = [
        validate_seed_case(
            case_root=case_root,
            seed=seed,
            frequency_heave_rao=frequency_heave_rao,
            discard_peak_cycles=args.discard_peak_cycles,
        )
        for seed, case_root in zip(seeds, case_roots)
    ]

    csv_path = write_seed_csv(output_root / "seed_sweep_metrics.csv", rows)
    figure = plot_seed_errors(output_root / "figures" / "seed_sweep_errors.png", rows)
    fit_errors = np.array([row["frequency_vs_time_fit_rms_l2_relative_error"] for row in rows], dtype=float)
    series_errors = np.array([row["frequency_vs_time_series_rms_l2_relative_error"] for row in rows], dtype=float)
    spectrum_errors = np.array([row["motion_spectrum_density_l2_relative_error"] for row in rows], dtype=float)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "hydrodynamic_dataset": hydro_path,
        "seed_count": len(seeds),
        "seeds": seeds,
        "radiation_residual_model": args.radiation_residual_model,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "radiation_passivity_correction": args.radiation_passivity_correction,
        "target_omega_rad_s": args.target_omega,
        "component_count": int(first_omega.size),
        "elapsed_seconds": timer.perf_counter() - start,
        "summary": {
            "frequency_vs_time_fit_rms_l2_relative_error": summarize(fit_errors),
            "frequency_vs_time_series_rms_l2_relative_error": summarize(series_errors),
            "motion_spectrum_density_l2_relative_error": summarize(spectrum_errors),
        },
        "rows": rows,
        "csv": csv_path,
        "figures": [figure],
    }
    metrics_path = write_metrics_json(output_root / "seed_sweep_metrics.json", metrics)

    print("Spectrum seed-sweep validation completed.")
    print(f"seed_count: {len(seeds)}")
    print(f"fit_rms_error_mean: {metrics['summary']['frequency_vs_time_fit_rms_l2_relative_error']['mean']:.6g}")
    print(f"time_series_rms_error_mean: {metrics['summary']['frequency_vs_time_series_rms_l2_relative_error']['mean']:.6g}")
    print(f"motion_spectrum_error_mean: {metrics['summary']['motion_spectrum_density_l2_relative_error']['mean']:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
