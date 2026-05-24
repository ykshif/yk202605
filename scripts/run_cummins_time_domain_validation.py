"""Validate the Cummins direct-convolution time-domain path on real BM10 data."""

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
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    fit_harmonic_amplitude,
    harmonic_amplitude_error,
    radiation_coefficients_from_discrete_irf,
    radiation_coefficients_from_irf,
    solve_rodm_time_domain_case,
)

from run_time_domain_reference_case_300 import (  # noqa: E402
    centerline_heave_time,
    default_dm_fem_root,
    build_default_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "cummins_bm10_validation"
DEFAULT_MULTI_FREQUENCY_HYDRO = Path("HydrodynamicData") / "Yoga" / "BM10_direaction0_full.nc"
BENCHMARK_300M_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_MULTI_FREQUENCY_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=BENCHMARK_300M_OMEGA)
    parser.add_argument("--frequency-index", type=int, default=None)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--cycles", type=float, default=12.0)
    parser.add_argument("--steps-per-cycle", type=int, default=45)
    parser.add_argument("--ramp-cycles", type=float, default=2.0)
    parser.add_argument("--discard-cycles", type=float, default=7.0)
    parser.add_argument("--memory-cycles", type=float, default=1.5)
    parser.add_argument("--wave-amplitude", type=float, default=1.0)
    parser.add_argument(
        "--damping-convention",
        choices=("physical", "wec_sim_bemio"),
        default="physical",
        help="Radiation damping convention used to build the IRF.",
    )
    parser.add_argument(
        "--infinite-added-mass-method",
        choices=("high_frequency", "ogilvie"),
        default="high_frequency",
    )
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default="clip_negative_eigenvalues",
        help="Correction applied to radiation damping before IRF generation.",
    )
    parser.add_argument("--added-mass-tail-count", type=int, default=3)
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default="none",
        help="Optional finite-band residual correction for regular-wave validation.",
    )
    parser.add_argument(
        "--radiation-frequency-window",
        choices=("none", "linear_tail", "cosine_tail"),
        default="none",
        help="Optional high-frequency damping taper before IRF generation.",
    )
    parser.add_argument("--radiation-window-start-omega", type=float, default=None)
    parser.add_argument("--radiation-window-stop-omega", type=float, default=None)
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default="rectangular",
        help="Discrete quadrature rule used for the radiation-memory convolution.",
    )
    parser.add_argument(
        "--skip-constant",
        action="store_true",
        help="Skip the constant-coefficient time-domain comparison run.",
    )
    return parser.parse_args()


def omega_values(path: Path) -> np.ndarray:
    """Read omega values from a hydrodynamic NetCDF file."""

    import xarray as xr

    dataset = xr.open_dataset(path)
    try:
        return np.asarray(dataset.omega.values, dtype=float).reshape(-1)
    finally:
        dataset.close()


def selected_frequency_index(values: np.ndarray, target_omega: float, override: int | None) -> int:
    """Return the requested or nearest frequency index."""

    if override is not None:
        if override < 0 or override >= values.size:
            raise ValueError("--frequency-index is outside the hydrodynamic omega grid")
        return int(override)
    return int(np.argmin(np.abs(values - target_omega)))


def representative_columns(count: int) -> tuple[int, int, int]:
    """Return bow/mid/stern-like column indices."""

    if count < 3:
        raise ValueError("heave array must contain at least three columns")
    return (0, count // 2, count - 1)


def write_representative_csv(path: Path, time: np.ndarray, heave: np.ndarray) -> Path:
    """Write representative Cummins heave histories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = representative_columns(heave.shape[1])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "x0", "x_mid", "x1"))
        for row_index, t in enumerate(time):
            writer.writerow([float(t), *(float(heave[row_index, col]) for col in columns)])
    return path


def plot_final_cycles(
    path: Path,
    time: np.ndarray,
    heave: np.ndarray,
    period: float,
    *,
    cycles: float = 4.0,
) -> Path:
    """Plot representative heave histories over the final cycles."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= time[-1] - cycles * period
    columns = representative_columns(heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for column, label, color in zip(columns, labels, colors):
        ax.plot(time[mask], heave[mask, column], color=color, linewidth=1.3, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heave displacement")
    ax.set_title("Cummins direct-convolution heave histories")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_midpoint_comparison(
    path: Path,
    time: np.ndarray,
    cummins_heave: np.ndarray,
    constant_heave: np.ndarray | None,
    period: float,
) -> Path:
    """Plot midpoint heave for Cummins and optional constant-coefficient runs."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= time[-1] - 4.0 * period
    mid = cummins_heave.shape[1] // 2
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if constant_heave is not None:
        ax.plot(
            time[mask],
            constant_heave[mask, mid],
            color="#1f77b4",
            linewidth=1.2,
            label="constant A/B",
        )
    ax.plot(
        time[mask],
        cummins_heave[mask, mid],
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label="Cummins convolution",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Midpoint heave displacement")
    ax.set_title("Midpoint heave comparison, final 4 cycles")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_amplitude_comparison(
    path: Path,
    frequency_global: np.ndarray,
    cummins_fit: np.ndarray,
    constant_fit: np.ndarray | None,
    *,
    retained_dofs_per_node: int,
) -> Path:
    """Plot centerline heave amplitudes from frequency and time-domain fits."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    freq_heave = centerline_heave_time(
        frequency_global.reshape(1, -1),
        retained_dofs_per_node=retained_dofs_per_node,
    )[0]
    cummins_heave = centerline_heave_time(
        cummins_fit.reshape(1, -1),
        retained_dofs_per_node=retained_dofs_per_node,
    )[0]
    x = np.linspace(0.0, 1.0, freq_heave.size)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(x, np.abs(freq_heave), color="#111111", linewidth=1.7, label="frequency A/B")
    if constant_fit is not None:
        constant_heave = centerline_heave_time(
            constant_fit.reshape(1, -1),
            retained_dofs_per_node=retained_dofs_per_node,
        )[0]
        ax.plot(
            x,
            np.abs(constant_heave),
            color="#1f77b4",
            linestyle=":",
            linewidth=1.5,
            label="constant time fit",
        )
    ax.plot(
        x,
        np.abs(cummins_heave),
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Cummins time fit",
    )
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave amplitude")
    ax.set_title("Frequency-domain and Cummins time-domain amplitude")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_norm_history(path: Path, x: np.ndarray, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> Path:
    """Plot a norm history for IRF or memory force."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if values.ndim == 3:
        y = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)
    else:
        y = np.linalg.norm(values, axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(x, y, color="#9467bd", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def relative_matrix_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return a Frobenius relative error."""

    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1.0e-30))


def main() -> int:
    args = parse_args()
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file
    if not hydro_path.is_absolute():
        hydro_path = data_root / hydro_path
    omega_grid = omega_values(hydro_path)
    frequency_index = selected_frequency_index(omega_grid, args.target_omega, args.frequency_index)
    omega = float(omega_grid[frequency_index])
    period = 2.0 * np.pi / omega
    time_step = period / args.steps_per_cycle
    duration = args.cycles * period
    ramp_time = args.ramp_cycles * period
    memory_duration = args.memory_cycles * period

    base_case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    case = replace(
        base_case,
        case_id=f"cummins_validation_{hydro_path.stem}",
        hydrodynamic_dataset=hydro_path,
        frequency_index=frequency_index,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"

    start = timer.perf_counter()
    frequency = solve_rodm_frequency_case(case)
    frequency_elapsed = timer.perf_counter() - start
    frequency_global = args.wave_amplitude * frequency.global_displacement.reshape(-1)

    constant = None
    constant_elapsed = None
    constant_fit = None
    constant_error = None
    constant_heave = None
    if not args.skip_constant:
        constant_config = TimeDomainSimulationConfig(
            time_step=time_step,
            duration=duration,
            wave_amplitude=args.wave_amplitude,
            ramp_time=ramp_time,
            radiation_model="constant",
        )
        start = timer.perf_counter()
        constant = solve_rodm_time_domain_case(case, constant_config)
        constant_elapsed = timer.perf_counter() - start
        constant_fit = fit_harmonic_amplitude(
            constant.global_displacement,
            constant.time,
            omega,
            start_time=args.discard_cycles * period,
        )
        constant_error = harmonic_amplitude_error(constant_fit, frequency_global)
        constant_heave = centerline_heave_time(
            constant.global_displacement,
            retained_dofs_per_node=case.retained_dofs_per_node,
        )
        np.save(output_root / "constant_global_displacement_time.npy", constant.global_displacement)

    cummins_config = TimeDomainSimulationConfig(
        time_step=time_step,
        duration=duration,
        wave_amplitude=args.wave_amplitude,
        ramp_time=ramp_time,
        radiation_model="direct_convolution",
        memory_duration=memory_duration,
        damping_convention=args.damping_convention,
        infinite_added_mass_method=args.infinite_added_mass_method,
        added_mass_tail_count=args.added_mass_tail_count,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_residual_model=args.radiation_residual_model,
        radiation_frequency_window=args.radiation_frequency_window,
        radiation_window_start_omega=args.radiation_window_start_omega,
        radiation_window_stop_omega=args.radiation_window_stop_omega,
        radiation_convolution_rule=args.radiation_convolution_rule,
    )
    start = timer.perf_counter()
    cummins = solve_rodm_time_domain_case(case, cummins_config)
    cummins_elapsed = timer.perf_counter() - start
    cummins_fit = fit_harmonic_amplitude(
        cummins.global_displacement,
        cummins.time,
        omega,
        start_time=args.discard_cycles * period,
    )
    cummins_error = harmonic_amplitude_error(cummins_fit, frequency_global)
    cummins_heave = centerline_heave_time(
        cummins.global_displacement,
        retained_dofs_per_node=case.retained_dofs_per_node,
    )

    added_from_irf = None
    damping_from_irf = None
    irf_added_error = None
    irf_damping_error = None
    corrected_added_error = None
    corrected_damping_error = None
    if cummins.radiation_irf is not None and cummins.added_mass_infinite is not None:
        added_from_irf, damping_from_irf = radiation_coefficients_from_irf(
            omega,
            cummins.radiation_irf,
            cummins.radiation_irf_time,
            added_mass_infinite=cummins.added_mass_infinite,
        )
        irf_added_error = relative_matrix_error(added_from_irf, frequency.added_mass)
        irf_damping_error = relative_matrix_error(damping_from_irf, frequency.radiation_damping)
        if (
            cummins.residual_added_mass is not None
            and cummins.residual_radiation_damping is not None
        ):
            discrete_added, discrete_damping = radiation_coefficients_from_discrete_irf(
                omega,
                cummins.radiation_irf,
                cummins.radiation_irf_time,
                added_mass_infinite=cummins.added_mass_infinite,
                convolution_rule=args.radiation_convolution_rule,
            )
            corrected_added_error = relative_matrix_error(
                discrete_added + cummins.residual_added_mass,
                frequency.added_mass,
            )
            corrected_damping_error = relative_matrix_error(
                discrete_damping + cummins.residual_radiation_damping,
                frequency.radiation_damping,
            )

    np.save(output_root / "time.npy", cummins.time)
    np.save(output_root / "cummins_global_displacement_time.npy", cummins.global_displacement)
    np.save(output_root / "cummins_master_displacement_time.npy", cummins.master_displacement)
    np.save(output_root / "cummins_memory_force_time.npy", cummins.memory_force)
    np.save(output_root / "cummins_centerline_heave_time.npy", cummins_heave)
    np.save(output_root / "cummins_fitted_global_amplitude.npy", cummins_fit)
    np.save(output_root / "frequency_global_amplitude.npy", frequency_global)
    if cummins.radiation_irf is not None:
        np.save(output_root / "radiation_irf.npy", cummins.radiation_irf)
        np.save(output_root / "radiation_irf_time.npy", cummins.radiation_irf_time)
    if cummins.added_mass_infinite is not None:
        np.save(output_root / "added_mass_infinite.npy", cummins.added_mass_infinite)
    if cummins.residual_added_mass is not None:
        np.save(output_root / "residual_added_mass.npy", cummins.residual_added_mass)
    if cummins.residual_radiation_damping is not None:
        np.save(output_root / "residual_radiation_damping.npy", cummins.residual_radiation_damping)
    if added_from_irf is not None:
        np.save(output_root / "selected_added_mass_from_irf.npy", added_from_irf)
        np.save(output_root / "selected_damping_from_irf.npy", damping_from_irf)

    csv_path = write_representative_csv(
        output_root / "cummins_representative_heave.csv",
        cummins.time,
        cummins_heave,
    )
    figures = [
        plot_final_cycles(
            figures_dir / "cummins_representative_heave_final_cycles.png",
            cummins.time,
            cummins_heave,
            period,
        ),
        plot_midpoint_comparison(
            figures_dir / "midpoint_heave_constant_vs_cummins.png",
            cummins.time,
            cummins_heave,
            constant_heave,
            period,
        ),
        plot_amplitude_comparison(
            figures_dir / "frequency_constant_cummins_heave_amplitude.png",
            frequency_global,
            cummins_fit,
            constant_fit,
            retained_dofs_per_node=case.retained_dofs_per_node,
        ),
        plot_norm_history(
            figures_dir / "cummins_memory_force_norm.png",
            cummins.time,
            cummins.memory_force,
            "Cummins radiation-memory force norm",
            "Time (s)",
            "Memory-force norm",
        ),
    ]
    if cummins.radiation_irf is not None:
        figures.append(
            plot_norm_history(
                figures_dir / "radiation_irf_norm.png",
                cummins.radiation_irf_time,
                cummins.radiation_irf,
                "Radiation IRF norm",
                "Memory time (s)",
                "IRF Frobenius norm",
            )
        )

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "case_id": case.case_id,
        "hydrodynamic_dataset": hydro_path,
        "target_omega_rad_s": args.target_omega,
        "frequency_index": frequency_index,
        "omega_rad_s": omega,
        "nearest_omega_difference": abs(omega - args.target_omega),
        "period_s": period,
        "time_step_s": time_step,
        "duration_s": duration,
        "cycles": args.cycles,
        "steps_per_cycle": args.steps_per_cycle,
        "ramp_cycles": args.ramp_cycles,
        "discard_cycles": args.discard_cycles,
        "memory_cycles": args.memory_cycles,
        "memory_duration_s": memory_duration,
        "wave_amplitude": args.wave_amplitude,
        "structural_reduction_method": case.structural_reduction_method,
        "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
        "infinite_added_mass_method": args.infinite_added_mass_method,
        "damping_convention": args.damping_convention,
        "radiation_passivity_correction": args.radiation_passivity_correction,
        "radiation_residual_model": args.radiation_residual_model,
        "radiation_frequency_window": args.radiation_frequency_window,
        "radiation_window_start_omega": args.radiation_window_start_omega,
        "radiation_window_stop_omega": args.radiation_window_stop_omega,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "added_mass_tail_count": args.added_mass_tail_count,
        "frequency_elapsed_seconds": frequency_elapsed,
        "constant_elapsed_seconds": constant_elapsed,
        "cummins_elapsed_seconds": cummins_elapsed,
        "constant_global_amplitude_error": constant_error,
        "cummins_global_amplitude_error": cummins_error,
        "irf_added_mass_relative_error_at_selected_omega": irf_added_error,
        "irf_damping_relative_error_at_selected_omega": irf_damping_error,
        "corrected_added_mass_relative_error_at_selected_omega": corrected_added_error,
        "corrected_damping_relative_error_at_selected_omega": corrected_damping_error,
        "time_samples": int(cummins.time.size),
        "global_displacement_shape": cummins.global_displacement.shape,
        "centerline_heave_shape": cummins_heave.shape,
        "representative_csv": csv_path,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "metrics.json", metrics)

    print("Cummins time-domain validation completed.")
    print(f"omega_rad_s: {omega:.8g}")
    print(f"frequency_index: {frequency_index}")
    if constant_error is not None:
        print(f"constant_l2_relative_error: {constant_error['l2_relative_error']:.6g}")
    print(f"cummins_l2_relative_error: {cummins_error['l2_relative_error']:.6g}")
    if irf_damping_error is not None:
        print(f"irf_damping_relative_error: {irf_damping_error:.6g}")
        print(f"irf_added_mass_relative_error: {irf_added_error:.6g}")
    if corrected_damping_error is not None:
        print(f"corrected_damping_relative_error: {corrected_damping_error:.6g}")
        print(f"corrected_added_mass_relative_error: {corrected_added_error:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
