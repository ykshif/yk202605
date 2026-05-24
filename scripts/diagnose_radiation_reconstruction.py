"""Diagnose how well a Cummins radiation kernel reconstructs A(omega), B(omega)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import csv
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    radiation_coefficients_from_discrete_irf,
    radiation_coefficients_from_irf,
    spectral_component_widths,
    wave_spectrum_density,
)
from offshore_energy_sim.time_domain.rodm_hydrodynamics import (  # noqa: E402
    _reduced_matrix_series,
    prepare_rodm_time_domain_hydrodynamic_terms,
)
from run_time_domain_reference_case_300 import build_default_case, default_dm_fem_root  # noqa: E402


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_spectrum_dense_88_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "rad_recon_dense88"
DEFAULT_TARGET_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=DEFAULT_TARGET_OMEGA)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=40)
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
    return parser.parse_args()


def weighted_relative_series_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    target = np.asarray(reference, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if residual.shape != target.shape or residual.shape[0] != w.size:
        raise ValueError("series and weights have incompatible shapes")
    numerator = np.sum(w * np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1) ** 2)
    denominator = np.sum(w * np.linalg.norm(target.reshape(target.shape[0], -1), axis=1) ** 2)
    return float(np.sqrt(numerator / max(float(denominator), 1.0e-30)))


def per_frequency_relative_error(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    residual = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    target = np.asarray(reference, dtype=float)
    numerator = np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
    denominator = np.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    return numerator / np.maximum(denominator, 1.0e-30)


def energy_band_mask(weights: np.ndarray, *, low: float = 0.05, high: float = 0.95) -> np.ndarray:
    positive = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(positive))
    if total <= 0.0:
        return np.ones_like(positive, dtype=bool)
    cumulative = np.cumsum(positive) / total
    return (cumulative >= low) & (cumulative <= high)


def dominant_diagonal_index(damping: np.ndarray) -> int:
    diagonal = np.abs(np.diagonal(damping, axis1=1, axis2=2))
    return int(np.argmax(np.mean(diagonal, axis=0)))


def write_error_csv(
    path: Path,
    omega: np.ndarray,
    spectrum_weights: np.ndarray,
    errors: dict[str, np.ndarray],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["omega_rad_s", "spectrum_weight", *errors.keys()])
        for index, omega_value in enumerate(omega):
            writer.writerow(
                [
                    float(omega_value),
                    float(spectrum_weights[index]),
                    *(float(values[index]) for values in errors.values()),
                ]
            )
    return path


def plot_ab_reconstruction(
    path: Path,
    omega: np.ndarray,
    original_added: np.ndarray,
    original_damping: np.ndarray,
    continuous_added: np.ndarray,
    continuous_damping: np.ndarray,
    discrete_added: np.ndarray,
    discrete_damping: np.ndarray,
    corrected_added: np.ndarray,
    corrected_damping: np.ndarray,
    *,
    dof: int,
    selected_omega: float,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True)
    axes[0].plot(omega, original_added[:, dof, dof], color="#111111", linewidth=1.5, label="original")
    axes[0].plot(omega, continuous_added[:, dof, dof], color="#1f77b4", linestyle=":", linewidth=1.2, label="continuous IRF")
    axes[0].plot(omega, discrete_added[:, dof, dof], color="#ff7f0e", linestyle="--", linewidth=1.2, label="discrete IRF")
    axes[0].plot(omega, corrected_added[:, dof, dof], color="#d62728", linestyle="-.", linewidth=1.2, label="with residual")
    axes[0].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[0].set_ylabel("A_ii(omega)")
    axes[0].set_title(f"Added-mass reconstruction, DOF {dof}")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False, ncols=2)

    axes[1].plot(omega, original_damping[:, dof, dof], color="#111111", linewidth=1.5, label="original")
    axes[1].plot(omega, continuous_damping[:, dof, dof], color="#1f77b4", linestyle=":", linewidth=1.2, label="continuous IRF")
    axes[1].plot(omega, discrete_damping[:, dof, dof], color="#ff7f0e", linestyle="--", linewidth=1.2, label="discrete IRF")
    axes[1].plot(omega, corrected_damping[:, dof, dof], color="#d62728", linestyle="-.", linewidth=1.2, label="with residual")
    axes[1].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("B_ii(omega)")
    axes[1].set_title(f"Radiation-damping reconstruction, DOF {dof}")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_error_curves(
    path: Path,
    omega: np.ndarray,
    weights: np.ndarray,
    errors: dict[str, np.ndarray],
    *,
    selected_omega: float,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    scaled_weight = weights / max(float(np.max(weights)), 1.0e-30)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.5), sharex=True)
    axes[0].plot(omega, errors["added_discrete"], color="#ff7f0e", linestyle="--", label="A discrete")
    axes[0].plot(omega, errors["added_corrected"], color="#d62728", linestyle="-.", label="A with residual")
    axes[0].plot(omega, scaled_weight, color="#2ca02c", linewidth=1.0, alpha=0.7, label="scaled wave weight")
    axes[0].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[0].set_ylabel("Relative Frobenius error")
    axes[0].set_title("Added-mass reconstruction error")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)

    axes[1].plot(omega, errors["damping_discrete"], color="#ff7f0e", linestyle="--", label="B discrete")
    axes[1].plot(omega, errors["damping_corrected"], color="#d62728", linestyle="-.", label="B with residual")
    axes[1].plot(omega, scaled_weight, color="#2ca02c", linewidth=1.0, alpha=0.7, label="scaled wave weight")
    axes[1].axvline(selected_omega, color="#777777", linewidth=0.9)
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("Relative Frobenius error")
    axes[1].set_title("Radiation-damping reconstruction error")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("direct-convolution hydrodynamic preprocessing did not produce an IRF")

    continuous_added, continuous_damping = radiation_coefficients_from_irf(
        omega,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
    )
    discrete_added, discrete_damping = radiation_coefficients_from_discrete_irf(
        omega,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
        convolution_rule=args.radiation_convolution_rule,
    )
    residual_added = (
        np.zeros_like(discrete_added[0])
        if terms.residual_added_mass is None
        else terms.residual_added_mass
    )
    residual_damping = (
        np.zeros_like(discrete_damping[0])
        if terms.residual_radiation_damping is None
        else terms.residual_radiation_damping
    )
    corrected_added = discrete_added + residual_added[np.newaxis, :, :]
    corrected_damping = discrete_damping + residual_damping[np.newaxis, :, :]

    density = wave_spectrum_density(
        omega,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=peak_period,
        gamma=args.peak_enhancement_factor,
    )
    weights = density * spectral_component_widths(omega)
    if np.sum(weights) <= 0.0:
        weights = np.ones_like(omega)
    weights = weights / np.sum(weights)
    band = energy_band_mask(weights)
    selected = int(np.argmin(np.abs(omega - selected_omega)))

    errors = {
        "added_continuous": per_frequency_relative_error(continuous_added, original_added),
        "damping_continuous": per_frequency_relative_error(continuous_damping, original_damping),
        "added_discrete": per_frequency_relative_error(discrete_added, original_added),
        "damping_discrete": per_frequency_relative_error(discrete_damping, original_damping),
        "added_corrected": per_frequency_relative_error(corrected_added, original_added),
        "damping_corrected": per_frequency_relative_error(corrected_damping, original_damping),
    }
    dof = dominant_diagonal_index(original_damping)
    figures = [
        plot_ab_reconstruction(
            output_root / "figures" / "radiation_ab_reconstruction_dominant_dof.png",
            omega,
            original_added,
            original_damping,
            continuous_added,
            continuous_damping,
            discrete_added,
            discrete_damping,
            corrected_added,
            corrected_damping,
            dof=dof,
            selected_omega=selected_omega,
        ),
        plot_error_curves(
            output_root / "figures" / "radiation_reconstruction_errors.png",
            omega,
            weights,
            errors,
            selected_omega=selected_omega,
        ),
    ]
    csv_path = write_error_csv(output_root / "radiation_reconstruction_errors.csv", omega, weights, errors)

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "hydrodynamic_dataset": hydro_path,
        "radiation_residual_model": args.radiation_residual_model,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "radiation_passivity_correction": args.radiation_passivity_correction,
        "selected_omega_rad_s": selected_omega,
        "dominant_diagonal_dof": dof,
        "component_count": int(omega.size),
        "memory_time_count": int(terms.radiation_irf_time.size),
        "memory_duration_s": float(terms.radiation_irf_time[-1]),
        "weighted_relative_errors": {
            "added_continuous": weighted_relative_series_error(continuous_added, original_added, weights),
            "damping_continuous": weighted_relative_series_error(continuous_damping, original_damping, weights),
            "added_discrete": weighted_relative_series_error(discrete_added, original_added, weights),
            "damping_discrete": weighted_relative_series_error(discrete_damping, original_damping, weights),
            "added_corrected": weighted_relative_series_error(corrected_added, original_added, weights),
            "damping_corrected": weighted_relative_series_error(corrected_damping, original_damping, weights),
        },
        "energy_band_relative_errors": {
            "omega_min": float(np.min(omega[band])),
            "omega_max": float(np.max(omega[band])),
            "added_discrete": weighted_relative_series_error(discrete_added[band], original_added[band], weights[band]),
            "damping_discrete": weighted_relative_series_error(discrete_damping[band], original_damping[band], weights[band]),
            "added_corrected": weighted_relative_series_error(corrected_added[band], original_added[band], weights[band]),
            "damping_corrected": weighted_relative_series_error(corrected_damping[band], original_damping[band], weights[band]),
        },
        "selected_frequency_relative_errors": {
            "added_discrete": float(errors["added_discrete"][selected]),
            "damping_discrete": float(errors["damping_discrete"][selected]),
            "added_corrected": float(errors["added_corrected"][selected]),
            "damping_corrected": float(errors["damping_corrected"][selected]),
        },
        "csv": csv_path,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "radiation_reconstruction_metrics.json", metrics)

    print("Radiation reconstruction diagnostics completed.")
    print(f"selected_omega_rad_s: {selected_omega:.6g}")
    print(f"added_weighted_error_discrete/corrected: {metrics['weighted_relative_errors']['added_discrete']:.6g} / {metrics['weighted_relative_errors']['added_corrected']:.6g}")
    print(f"damping_weighted_error_discrete/corrected: {metrics['weighted_relative_errors']['damping_discrete']:.6g} / {metrics['weighted_relative_errors']['damping_corrected']:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
