"""Compare spectrum-driven time-domain motion spectra with frequency-domain RAOs."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset, prepare_hydrodynamic_terms  # noqa: E402
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.structure import calculate_node_positions, prepare_structural_reduction  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_multi_harmonic_amplitudes,
    relative_l2_error,
    spectral_component_widths,
    wave_spectrum_density,
    zero_mean_rms,
)

from run_time_domain_reference_case_300 import (  # noqa: E402
    build_default_case,
    centerline_heave_time,
    default_dm_fem_root,
)


DEFAULT_CASE_ROOT = (
    REPO_ROOT
    / "results"
    / "time_domain"
    / "hydrodynamic_extrapolation_dm10_mesh2"
    / "time_domain_extrapolated"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "spectrum_frequency_time_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--discard-seconds", type=float, default=None)
    parser.add_argument("--discard-peak-cycles", type=float, default=5.0)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    return parser.parse_args()


def load_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def representative_columns(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("centerline heave must contain at least three columns")
    return (0, count // 2, count - 1)


def selected_frequency_indices(dataset_omega: np.ndarray, component_omega: np.ndarray) -> np.ndarray:
    indices = np.array([int(np.argmin(np.abs(dataset_omega - value))) for value in component_omega], dtype=int)
    mismatch = np.max(np.abs(dataset_omega[indices] - component_omega))
    if mismatch > 1.0e-10:
        raise ValueError(f"component omega values do not match dataset omega grid; max mismatch={mismatch}")
    return indices


def solve_centerline_frequency_rao(
    *,
    hydro_path: Path,
    data_root: Path,
    component_omega: np.ndarray,
    reversed_hydro: bool,
) -> np.ndarray:
    """Solve frequency-domain centerline heave RAOs on the component grid."""

    case = build_default_case(
        data_root,
        reversed_hydro=reversed_hydro,
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
        dataset_omega = np.asarray(dataset.omega.values, dtype=float).reshape(-1)
        frequency_indices = selected_frequency_indices(dataset_omega, component_omega)
        structural = prepare_structural_reduction(case, master_nodes)
        heave_rao = []
        for frequency_index in frequency_indices:
            frequency_case = replace(case, frequency_index=int(frequency_index))
            hydrodynamic = prepare_hydrodynamic_terms(frequency_case, dataset)
            effective_mass = hydrodynamic.added_mass + structural.reduced_mass
            effective_damping = hydrodynamic.radiation_damping
            effective_stiffness = hydrodynamic.hydrostatic_stiffness + structural.reduced_stiffness
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
            heave = centerline_heave_time(
                global_displacement.reshape(1, -1),
                retained_dofs_per_node=case.retained_dofs_per_node,
            )[0]
            heave_rao.append(heave)
    finally:
        dataset.close()
    return np.stack(heave_rao, axis=0)


def motion_spectral_density(
    component_amplitude: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    widths = spectral_component_widths(omega)
    return 0.5 * np.abs(component_amplitude) ** 2 / widths[:, np.newaxis]


def write_centerline_rms_csv(
    path: Path,
    frequency_rms: np.ndarray,
    time_fit_rms: np.ndarray,
    time_series_rms: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, frequency_rms.size)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "centerline_index",
                "x_over_L",
                "frequency_rms",
                "time_fit_rms",
                "time_series_rms",
                "time_fit_minus_frequency",
                "time_series_minus_frequency",
            )
        )
        for index, (freq, fit, series) in enumerate(zip(frequency_rms, time_fit_rms, time_series_rms)):
            writer.writerow(
                [
                    index,
                    float(x[index]),
                    float(freq),
                    float(fit),
                    float(series),
                    float(fit - freq),
                    float(series - freq),
                ]
            )
    return path


def plot_motion_spectrum_comparison(
    path: Path,
    omega: np.ndarray,
    frequency_density: np.ndarray,
    time_density: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = representative_columns(frequency_density.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.2), sharex=True)
    for ax, column, label in zip(axes, columns, labels):
        ax.plot(omega, frequency_density[:, column], color="#111111", linewidth=1.5, label="frequency-domain")
        ax.plot(omega, time_density[:, column], color="#d62728", linestyle="--", linewidth=1.2, label="time-domain fit")
        ax.set_ylabel("S_z(omega)")
        ax.set_title(label)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Angular frequency (rad/s)")
    fig.suptitle("Centerline heave motion-spectrum comparison", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_centerline_rms_comparison(
    path: Path,
    frequency_rms: np.ndarray,
    time_fit_rms: np.ndarray,
    time_series_rms: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, frequency_rms.size)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(x, frequency_rms, color="#111111", linewidth=1.7, label="frequency-domain spectrum integral")
    ax.plot(x, time_fit_rms, color="#d62728", linestyle="--", linewidth=1.4, label="time-domain harmonic fit")
    ax.plot(x, time_series_rms, color="#1f77b4", linestyle=":", linewidth=1.4, label="time-domain time-series RMS")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RMS")
    ax.set_title("Frequency-domain and time-domain RMS comparison")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_wave_and_motion_spectrum(
    path: Path,
    omega: np.ndarray,
    wave_density: np.ndarray,
    frequency_density: np.ndarray,
    time_density: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mid = frequency_density.shape[1] // 2
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    axes[0].plot(omega, wave_density, color="#2ca02c", linewidth=1.5)
    axes[0].set_ylabel("S_eta(omega)")
    axes[0].set_title("Input wave spectrum")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].plot(omega, frequency_density[:, mid], color="#111111", linewidth=1.5, label="frequency-domain")
    axes[1].plot(omega, time_density[:, mid], color="#d62728", linestyle="--", linewidth=1.2, label="time-domain fit")
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("S_z(omega)")
    axes[1].set_title("Midpoint heave response spectrum")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    case_root = args.case_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = load_json(case_root / "metrics.json")
    hydro_path = Path(str(metrics["hydrodynamic_dataset"]))
    data_root = default_dm_fem_root(args.data_root)
    time = load_array(case_root, "time.npy")
    omega = load_array(case_root, "wave_component_omega.npy")
    wave_amplitude = load_array(case_root, "wave_component_amplitude.npy")
    heave = load_array(case_root, "centerline_heave_time.npy")

    discard_seconds = (
        args.discard_seconds
        if args.discard_seconds is not None
        else args.discard_peak_cycles * float(metrics["peak_period_s"])
    )
    mask = time >= time[0] + discard_seconds
    if np.count_nonzero(mask) <= 2 * omega.size + 1:
        raise ValueError("not enough post-discard samples for multi-harmonic fit")
    fit_start = float(time[mask][0])

    start = timer.perf_counter()
    frequency_heave_rao = solve_centerline_frequency_rao(
        hydro_path=hydro_path,
        data_root=data_root,
        component_omega=omega,
        reversed_hydro=args.hydro_node_order == "reversed",
    )
    frequency_elapsed = timer.perf_counter() - start

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
    wave_density = wave_spectrum_density(
        omega,
        spectrum_type=str(metrics["spectrum_type"]),
        significant_wave_height=float(metrics["significant_wave_height"]),
        peak_period=float(metrics["peak_period_s"]),
    )

    figures = [
        plot_motion_spectrum_comparison(
            output_root / "figures" / "motion_spectrum_frequency_vs_time.png",
            omega,
            frequency_density,
            time_density,
        ),
        plot_centerline_rms_comparison(
            output_root / "figures" / "centerline_rms_frequency_vs_time.png",
            frequency_rms,
            time_fit_rms,
            time_series_rms,
        ),
        plot_wave_and_motion_spectrum(
            output_root / "figures" / "wave_and_midpoint_motion_spectrum.png",
            omega,
            wave_density,
            frequency_density,
            time_density,
        ),
    ]
    csv_path = write_centerline_rms_csv(
        output_root / "centerline_rms_frequency_vs_time.csv",
        frequency_rms,
        time_fit_rms,
        time_series_rms,
    )
    np.save(output_root / "frequency_centerline_heave_rao.npy", frequency_heave_rao)
    np.save(output_root / "time_centerline_heave_components.npy", time_heave_components)
    np.save(output_root / "frequency_motion_spectrum_density.npy", frequency_density)
    np.save(output_root / "time_motion_spectrum_density.npy", time_density)

    comparison = {
        "frequency_vs_time_fit_rms_l2_relative_error": relative_l2_error(time_fit_rms, frequency_rms),
        "frequency_vs_time_series_rms_l2_relative_error": relative_l2_error(time_series_rms, frequency_rms),
        "motion_spectrum_density_l2_relative_error": relative_l2_error(time_density, frequency_density),
        "frequency_rms_max": float(np.max(frequency_rms)),
        "time_fit_rms_max": float(np.max(time_fit_rms)),
        "time_series_rms_max": float(np.max(time_series_rms)),
        "frequency_rms_mean": float(np.mean(frequency_rms)),
        "time_fit_rms_mean": float(np.mean(time_fit_rms)),
        "time_series_rms_mean": float(np.mean(time_series_rms)),
    }
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "case_root": case_root,
        "hydrodynamic_dataset": hydro_path,
        "component_count": int(omega.size),
        "discard_seconds": discard_seconds,
        "fit_start_time_s": fit_start,
        "frequency_solve_elapsed_seconds": frequency_elapsed,
        "comparison": comparison,
        "centerline_csv": csv_path,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "frequency_time_motion_spectrum_metrics.json", summary)

    print("Spectrum frequency/time response comparison completed.")
    print(f"frequency_vs_time_fit_rms_l2_relative_error: {comparison['frequency_vs_time_fit_rms_l2_relative_error']:.6g}")
    print(f"frequency_vs_time_series_rms_l2_relative_error: {comparison['frequency_vs_time_series_rms_l2_relative_error']:.6g}")
    print(f"motion_spectrum_density_l2_relative_error: {comparison['motion_spectrum_density_l2_relative_error']:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
