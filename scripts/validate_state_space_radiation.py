"""Validate state-space radiation approximation against Cummins direct convolution."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import json
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
    evaluate_state_space_radiation_kernel,
    fit_era_state_space_radiation,
    fit_common_pole_state_space_radiation,
    simulate_era_memory_force,
    simulate_state_space_memory_force,
    state_space_radiation_coefficients,
)
from run_time_domain_reference_case_300 import build_default_case, default_dm_fem_root  # noqa: E402


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_spectrum_dense_88_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "state_space_radiation_dense88"
DEFAULT_TARGET_OMEGA = 0.4157
DEFAULT_VELOCITY_CASE = (
    REPO_ROOT
    / "results"
    / "time_domain"
    / "num_sens_dense88_step_focus"
    / "cases"
    / "mem_2_steps_60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--fit-method", choices=("common_pole", "era"), default="era")
    parser.add_argument("--state-order", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=0.0)
    parser.add_argument("--era-block-rows", type=int, default=None)
    parser.add_argument("--era-block-cols", type=int, default=None)
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
    return parser.parse_args()


def relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(predicted) - np.asarray(reference)) / max(float(np.linalg.norm(reference)), 1.0e-30))


def weighted_relative_series_error(predicted: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    target = np.asarray(reference, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    numerator = np.sum(w * np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1) ** 2)
    denominator = np.sum(w * np.linalg.norm(target.reshape(target.shape[0], -1), axis=1) ** 2)
    return float(np.sqrt(numerator / max(float(denominator), 1.0e-30)))


def dominant_diagonal_index(damping: np.ndarray) -> int:
    diagonal = np.abs(np.diagonal(damping, axis1=1, axis2=2))
    return int(np.argmax(np.mean(diagonal, axis=0)))


def load_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def direct_convolution_memory_history(
    velocity: np.ndarray,
    kernel: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    force = np.zeros_like(velocity)
    dt = float(np.diff(time)[0])
    for index in range(1, time.size):
        force[index] = direct_convolution_memory_force(
            velocity,
            kernel,
            dt,
            index,
            convolution_rule="trapezoidal",
        )
    return force


def plot_kernel_norm(
    path: Path,
    time: np.ndarray,
    direct_kernel: np.ndarray,
    state_kernel: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    direct_norm = np.linalg.norm(direct_kernel.reshape(direct_kernel.shape[0], -1), axis=1)
    state_norm = np.linalg.norm(state_kernel.reshape(state_kernel.shape[0], -1), axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(time, direct_norm, color="#111111", linewidth=1.5, label="Cummins IRF")
    ax.plot(time, state_norm, color="#d62728", linestyle="--", linewidth=1.2, label="state-space fit")
    ax.set_xlabel("Memory time (s)")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Radiation-kernel state-space fit")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_ab_comparison(
    path: Path,
    omega: np.ndarray,
    original_added: np.ndarray,
    original_damping: np.ndarray,
    direct_added: np.ndarray,
    direct_damping: np.ndarray,
    state_added: np.ndarray,
    state_damping: np.ndarray,
    *,
    dof: int,
    selected_omega: float,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True)
    axes[0].plot(omega, original_added[:, dof, dof], color="#111111", linewidth=1.5, label="original")
    axes[0].plot(omega, direct_added[:, dof, dof], color="#1f77b4", linestyle=":", linewidth=1.2, label="direct IRF")
    axes[0].plot(omega, state_added[:, dof, dof], color="#d62728", linestyle="--", linewidth=1.2, label="state-space")
    axes[0].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[0].set_ylabel("A_ii(omega)")
    axes[0].set_title(f"State-space added-mass reconstruction, DOF {dof}")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)

    axes[1].plot(omega, original_damping[:, dof, dof], color="#111111", linewidth=1.5, label="original")
    axes[1].plot(omega, direct_damping[:, dof, dof], color="#1f77b4", linestyle=":", linewidth=1.2, label="direct IRF")
    axes[1].plot(omega, state_damping[:, dof, dof], color="#d62728", linestyle="--", linewidth=1.2, label="state-space")
    axes[1].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("B_ii(omega)")
    axes[1].set_title(f"State-space radiation-damping reconstruction, DOF {dof}")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_memory_force_norm(
    path: Path,
    time: np.ndarray,
    direct_force: np.ndarray,
    state_force: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    direct_norm = np.linalg.norm(direct_force, axis=1)
    state_norm = np.linalg.norm(state_force, axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(time, direct_norm, color="#111111", linewidth=1.3, label="direct convolution")
    ax.plot(time, state_norm, color="#d62728", linestyle="--", linewidth=1.1, label="state-space")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radiation-memory force norm")
    ax.set_title("State-space memory-force comparison")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


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

    if args.fit_method == "era":
        model = fit_era_state_space_radiation(
            terms.radiation_irf_time,
            terms.radiation_irf,
            order=args.state_order,
            block_rows=args.era_block_rows,
            block_cols=args.era_block_cols,
        )
        state_kernel = evaluate_era_radiation_kernel(model, terms.radiation_irf_time.size)
    else:
        model = fit_common_pole_state_space_radiation(
            terms.radiation_irf_time,
            terms.radiation_irf,
            order=args.state_order,
            ridge_alpha=args.ridge_alpha,
        )
        state_kernel = evaluate_state_space_radiation_kernel(model, terms.radiation_irf_time)
    direct_added, direct_damping = radiation_coefficients_from_discrete_irf(
        omega,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
        convolution_rule=args.radiation_convolution_rule,
    )
    if args.fit_method == "era":
        state_added, state_damping = radiation_coefficients_from_discrete_irf(
            omega,
            state_kernel,
            terms.radiation_irf_time,
            added_mass_infinite=terms.added_mass_infinite,
            convolution_rule=args.radiation_convolution_rule,
        )
    else:
        state_added, state_damping = state_space_radiation_coefficients(
            model,
            omega,
            added_mass_infinite=terms.added_mass_infinite,
        )
    residual_added = np.zeros_like(state_added[0]) if terms.residual_added_mass is None else terms.residual_added_mass
    residual_damping = (
        np.zeros_like(state_damping[0])
        if terms.residual_radiation_damping is None
        else terms.residual_radiation_damping
    )
    direct_added_corrected = direct_added + residual_added[np.newaxis, :, :]
    direct_damping_corrected = direct_damping + residual_damping[np.newaxis, :, :]
    state_added_corrected = state_added + residual_added[np.newaxis, :, :]
    state_damping_corrected = state_damping + residual_damping[np.newaxis, :, :]

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

    figures = [
        plot_kernel_norm(
            output_root / "figures" / "state_space_kernel_norm.png",
            terms.radiation_irf_time,
            terms.radiation_irf,
            state_kernel,
        ),
        plot_ab_comparison(
            output_root / "figures" / "state_space_ab_reconstruction.png",
            omega,
            original_added,
            original_damping,
            direct_added_corrected,
            direct_damping_corrected,
            state_added_corrected,
            state_damping_corrected,
            dof=dof,
            selected_omega=selected_omega,
        ),
    ]

    memory_force_metrics = None
    velocity_case_root = args.velocity_case_root
    if velocity_case_root is not None and velocity_case_root.exists():
        velocity = load_array(velocity_case_root, "master_velocity_time.npy")
        velocity_time = load_array(velocity_case_root, "time.npy")
        sample_count = min(int(args.memory_force_samples), velocity_time.size)
        velocity = velocity[:sample_count]
        velocity_time = velocity_time[:sample_count]
        direct_memory = direct_convolution_memory_history(
            velocity,
            terms.radiation_irf,
            velocity_time,
        )
        if args.fit_method == "era":
            state_memory = simulate_era_memory_force(velocity, velocity_time, model)
        else:
            state_memory = simulate_state_space_memory_force(velocity, velocity_time, model)
        memory_force_metrics = {
            "velocity_case_root": velocity_case_root,
            "sample_count": sample_count,
            "state_vs_direct_memory_force_l2_relative_error": relative_l2(state_memory, direct_memory),
            "direct_memory_force_rms": float(np.sqrt(np.mean(np.linalg.norm(direct_memory, axis=1) ** 2))),
            "state_memory_force_rms": float(np.sqrt(np.mean(np.linalg.norm(state_memory, axis=1) ** 2))),
        }
        np.save(output_root / "direct_memory_force_sample.npy", direct_memory)
        np.save(output_root / "state_space_memory_force_sample.npy", state_memory)
        figures.append(
            plot_memory_force_norm(
                output_root / "figures" / "state_space_memory_force_norm.png",
                velocity_time,
                direct_memory,
                state_memory,
            )
        )

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "hydrodynamic_dataset": hydro_path,
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "fit_method": args.fit_method,
        "state_space_order": args.state_order,
        "ridge_alpha": args.ridge_alpha,
        "era_block_rows": args.era_block_rows,
        "era_block_cols": args.era_block_cols,
        "state_space_model": model.to_dict(),
        "selected_omega_rad_s": selected_omega,
        "memory_cycles": args.memory_cycles,
        "steps_per_peak_cycle": args.steps_per_peak_cycle,
        "memory_time_count": int(terms.radiation_irf_time.size),
        "memory_duration_s": float(terms.radiation_irf_time[-1]),
        "component_count": int(omega.size),
        "dominant_diagonal_dof": dof,
        "kernel_fit": {
            "l2_relative_error": model.fit_l2_relative_error,
            "peak_relative_error": getattr(model, "fit_peak_relative_error", None),
        },
        "weighted_frequency_reconstruction_errors": {
            "direct_added_corrected": weighted_relative_series_error(direct_added_corrected, original_added, weights),
            "direct_damping_corrected": weighted_relative_series_error(direct_damping_corrected, original_damping, weights),
            "state_added_corrected": weighted_relative_series_error(state_added_corrected, original_added, weights),
            "state_damping_corrected": weighted_relative_series_error(state_damping_corrected, original_damping, weights),
            "state_vs_direct_added": weighted_relative_series_error(state_added, direct_added, weights),
            "state_vs_direct_damping": weighted_relative_series_error(state_damping, direct_damping, weights),
        },
        "selected_frequency_errors": {
            "state_added_corrected": relative_l2(state_added_corrected[selected], original_added[selected]),
            "state_damping_corrected": relative_l2(state_damping_corrected[selected], original_damping[selected]),
            "state_vs_direct_added": relative_l2(state_added[selected], direct_added[selected]),
            "state_vs_direct_damping": relative_l2(state_damping[selected], direct_damping[selected]),
        },
        "memory_force_metrics": memory_force_metrics,
        "elapsed_seconds": timer.perf_counter() - start,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "state_space_radiation_metrics.json", metrics)

    print("State-space radiation validation completed.")
    print(f"fit_method: {args.fit_method}")
    print(f"state_space_order: {args.state_order}")
    print(f"kernel_fit_l2_relative_error: {model.fit_l2_relative_error:.6g}")
    print(
        "state_vs_direct_added/damping weighted errors: "
        f"{metrics['weighted_frequency_reconstruction_errors']['state_vs_direct_added']:.6g} / "
        f"{metrics['weighted_frequency_reconstruction_errors']['state_vs_direct_damping']:.6g}"
    )
    if memory_force_metrics is not None:
        print(
            "state_vs_direct_memory_force_l2_relative_error: "
            f"{memory_force_metrics['state_vs_direct_memory_force_l2_relative_error']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
