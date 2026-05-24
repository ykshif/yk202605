"""Run a RODM time-domain case with spectrum or external-force excitation."""

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

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain import TimeDomainSimulationConfig, wave_spectrum_density  # noqa: E402
from offshore_energy_sim.time_domain.solver import solve_rodm_time_domain_case  # noqa: E402

from run_time_domain_reference_case_300 import (  # noqa: E402
    centerline_heave_time,
    default_dm_fem_root,
    build_default_case,
)


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "excitation_case_dm10_mesh2"
BENCHMARK_300M_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=BENCHMARK_300M_OMEGA)
    parser.add_argument("--frequency-index", type=int, default=None)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument(
        "--excitation-model",
        choices=("wave_spectrum", "external_force"),
        default="wave_spectrum",
    )
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--peak-period", type=float, default=None)
    parser.add_argument(
        "--spectrum-type",
        choices=("jonswap", "pierson_moskowitz"),
        default="jonswap",
    )
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--spectrum-seed", type=int, default=20260522)
    parser.add_argument("--external-force-csv", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--peak-cycles", type=float, default=40.0)
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=80)
    parser.add_argument("--ramp-cycles", type=float, default=2.0)
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
        "--radiation-frequency-window",
        choices=("none", "linear_tail", "cosine_tail"),
        default="none",
    )
    parser.add_argument("--radiation-window-start-omega", type=float, default=None)
    parser.add_argument("--radiation-window-stop-omega", type=float, default=None)
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default="none",
    )
    return parser.parse_args()


def omega_values(path: Path) -> np.ndarray:
    import xarray as xr

    dataset = xr.open_dataset(path)
    try:
        return np.asarray(dataset.omega.values, dtype=float).reshape(-1)
    finally:
        dataset.close()


def selected_frequency_index(values: np.ndarray, target_omega: float, override: int | None) -> int:
    if override is not None:
        if override < 0 or override >= values.size:
            raise ValueError("--frequency-index is outside the hydrodynamic omega grid")
        return int(override)
    return int(np.argmin(np.abs(values - target_omega)))


def load_external_force_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with ``time_s`` in column 1 and force columns afterwards."""

    with path.open("r", encoding="utf-8-sig") as stream:
        sample = stream.readline()
    has_header = any(char.isalpha() for char in sample)
    if has_header:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        names = data.dtype.names or ()
        if len(names) < 2:
            raise ValueError("external force CSV must contain time and force columns")
        time = np.asarray(data[names[0]], dtype=float)
        force = np.column_stack([np.asarray(data[name], dtype=float) for name in names[1:]])
    else:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("external force CSV must contain time and force columns")
        time = data[:, 0]
        force = data[:, 1:]
    return time, force


def representative_columns(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("heave array must contain at least three columns")
    return (0, count // 2, count - 1)


def write_representative_csv(path: Path, time: np.ndarray, heave: np.ndarray) -> Path:
    columns = representative_columns(heave.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "x0", "x_mid", "x1"))
        for row_index, t in enumerate(time):
            writer.writerow([float(t), *(float(heave[row_index, col]) for col in columns)])
    return path


def plot_representative_heave(path: Path, time: np.ndarray, heave: np.ndarray, *, window_s: float) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= max(time[0], time[-1] - window_s)
    columns = representative_columns(heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for column, label in zip(columns, labels):
        ax.plot(time[mask], heave[mask, column], linewidth=1.1, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heave displacement")
    ax.set_title("Time-domain representative heave histories")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_wave_elevation(path: Path, time: np.ndarray, wave_elevation: np.ndarray, *, window_s: float) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= max(time[0], time[-1] - window_s)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(time[mask], wave_elevation[mask], color="#1f77b4", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wave elevation (m)")
    ax.set_title("Synthesized wave elevation")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_force_norm(path: Path, time: np.ndarray, force: np.ndarray, *, window_s: float) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= max(time[0], time[-1] - window_s)
    force_norm = np.linalg.norm(force, axis=1)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(time[mask], force_norm[mask], color="#d62728", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Excitation-force norm")
    ax.set_title("Excitation force history")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_spectrum(path: Path, omega: np.ndarray, density: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(omega, density, color="#2ca02c", linewidth=1.4)
    ax.set_xlabel("Angular frequency (rad/s)")
    ax.set_ylabel("S(omega) (m^2 s/rad)")
    ax.set_title("Wave spectrum on BEM frequency grid")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    omega_grid = omega_values(hydro_path)
    frequency_index = selected_frequency_index(omega_grid, args.target_omega, args.frequency_index)
    selected_omega = float(omega_grid[frequency_index])
    selected_period = 2.0 * np.pi / selected_omega
    peak_period = args.peak_period or selected_period
    external_time = None
    external_force = None
    if args.excitation_model == "external_force":
        if args.external_force_csv is None:
            raise ValueError("--external-force-csv is required for external_force excitation")
        external_time, external_force = load_external_force_csv(args.external_force_csv)

    if args.time_step is not None:
        time_step = args.time_step
    elif external_time is not None:
        time_step = float(np.median(np.diff(external_time)))
    else:
        time_step = peak_period / args.steps_per_peak_cycle
    if args.duration is not None:
        duration = args.duration
    elif external_time is not None:
        duration = float(external_time[-1] - external_time[0])
    else:
        duration = args.peak_cycles * peak_period

    case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    case = replace(
        case,
        case_id=f"time_domain_{args.excitation_model}_{hydro_path.stem}",
        hydrodynamic_dataset=hydro_path,
        frequency_index=frequency_index,
    )
    config = TimeDomainSimulationConfig(
        time_step=time_step,
        duration=duration,
        excitation_model=args.excitation_model,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=peak_period,
        peak_enhancement_factor=args.peak_enhancement_factor,
        spectrum_seed=args.spectrum_seed,
        external_force_time=external_time,
        external_force=external_force,
        ramp_time=args.ramp_cycles * peak_period,
        radiation_model="direct_convolution",
        memory_duration=args.memory_cycles * peak_period,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_convolution_rule=args.radiation_convolution_rule,
        radiation_frequency_window=args.radiation_frequency_window,
        radiation_window_start_omega=args.radiation_window_start_omega,
        radiation_window_stop_omega=args.radiation_window_stop_omega,
        radiation_residual_model=args.radiation_residual_model,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    start = timer.perf_counter()
    result = solve_rodm_time_domain_case(case, config)
    elapsed = timer.perf_counter() - start
    heave = centerline_heave_time(
        result.global_displacement,
        retained_dofs_per_node=case.retained_dofs_per_node,
    )

    np.save(output_root / "time.npy", result.time)
    np.save(output_root / "global_displacement_time.npy", result.global_displacement)
    np.save(output_root / "master_displacement_time.npy", result.master_displacement)
    np.save(output_root / "master_velocity_time.npy", result.master_velocity)
    np.save(output_root / "master_acceleration_time.npy", result.master_acceleration)
    np.save(output_root / "centerline_heave_time.npy", heave)
    np.save(output_root / "memory_force_time.npy", result.memory_force)
    np.save(output_root / "excitation_force_time.npy", result.excitation_force)
    if result.wave_elevation is not None:
        np.save(output_root / "wave_elevation_time.npy", result.wave_elevation)
    if result.wave_component_omega is not None:
        np.save(output_root / "wave_component_omega.npy", result.wave_component_omega)
        np.save(output_root / "wave_component_amplitude.npy", result.wave_component_amplitude)
        np.save(output_root / "wave_component_phase.npy", result.wave_component_phase)

    figures_dir = output_root / "figures"
    window_s = min(duration, max(4.0 * peak_period, 120.0))
    figures = [
        plot_representative_heave(
            figures_dir / "representative_heave_histories.png",
            result.time,
            heave,
            window_s=window_s,
        ),
        plot_force_norm(
            figures_dir / "excitation_force_norm.png",
            result.time,
            result.excitation_force,
            window_s=window_s,
        ),
    ]
    if result.wave_elevation is not None:
        figures.append(
            plot_wave_elevation(
                figures_dir / "wave_elevation.png",
                result.time,
                result.wave_elevation,
                window_s=window_s,
            )
        )
        density = wave_spectrum_density(
            result.wave_component_omega,
            spectrum_type=args.spectrum_type,
            significant_wave_height=args.significant_wave_height,
            peak_period=peak_period,
            gamma=args.peak_enhancement_factor,
        )
        figures.append(plot_spectrum(figures_dir / "wave_spectrum.png", result.wave_component_omega, density))
    csv_path = write_representative_csv(output_root / "representative_heave.csv", result.time, heave)

    heave_rms = np.sqrt(np.mean(heave**2, axis=0))
    force_norm = np.linalg.norm(result.excitation_force, axis=1)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "case_id": case.case_id,
        "hydrodynamic_dataset": hydro_path,
        "excitation_model": args.excitation_model,
        "selected_omega_rad_s": selected_omega,
        "frequency_index": frequency_index,
        "time_step_s": time_step,
        "duration_s": duration,
        "time_samples": int(result.time.size),
        "peak_period_s": peak_period,
        "significant_wave_height": args.significant_wave_height,
        "spectrum_type": args.spectrum_type,
        "spectrum_seed": args.spectrum_seed,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "radiation_frequency_window": args.radiation_frequency_window,
        "radiation_residual_model": args.radiation_residual_model,
        "elapsed_seconds": elapsed,
        "global_displacement_shape": result.global_displacement.shape,
        "centerline_heave_shape": heave.shape,
        "centerline_heave_rms_max": float(np.max(heave_rms)),
        "centerline_heave_abs_max": float(np.max(np.abs(heave))),
        "excitation_force_norm_rms": float(np.sqrt(np.mean(force_norm**2))),
        "representative_csv": csv_path,
        "figures": figures,
    }
    if result.wave_elevation is not None:
        metrics.update(
            {
                "wave_elevation_rms": float(np.sqrt(np.mean(result.wave_elevation**2))),
                "wave_elevation_abs_max": float(np.max(np.abs(result.wave_elevation))),
            }
        )
    metrics_path = write_metrics_json(output_root / "metrics.json", metrics)

    print("Time-domain excitation case completed.")
    print(f"excitation_model: {args.excitation_model}")
    print(f"time_samples: {result.time.size}")
    print(f"centerline_heave_rms_max: {metrics['centerline_heave_rms_max']:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
