"""Design a spectrum-focused frequency grid for JONSWAP time-domain validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    spectral_component_widths,
    wave_spectrum_density,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "spectrum_frequency_grid_design"
DEFAULT_PEAK_PERIOD = 15.11471086644115


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--peak-period", type=float, default=DEFAULT_PEAK_PERIOD)
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--omega-min", type=float, default=0.05)
    parser.add_argument("--omega-max", type=float, default=2.0)
    parser.add_argument("--include-omega", type=float, action="append", default=[2.0 * np.pi / DEFAULT_PEAK_PERIOD])
    parser.add_argument(
        "--profile",
        choices=("balanced", "fine_peak"),
        default="balanced",
        help="Predefined segmented grid profile.",
    )
    return parser.parse_args()


def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    count = int(np.floor((stop - start) / step + 0.5))
    values = start + step * np.arange(count + 1, dtype=float)
    return values[values <= stop + 1.0e-10]


def segmented_grid(profile: str) -> np.ndarray:
    if profile == "balanced":
        segments = (
            (0.05, 0.25, 0.025),
            (0.25, 0.35, 0.0125),
            (0.35, 0.50, 0.0075),
            (0.50, 0.80, 0.015),
            (0.80, 1.20, 0.025),
            (1.20, 2.00, 0.05),
        )
    elif profile == "fine_peak":
        segments = (
            (0.05, 0.25, 0.025),
            (0.25, 0.34, 0.010),
            (0.34, 0.52, 0.005),
            (0.52, 0.80, 0.010),
            (0.80, 1.20, 0.025),
            (1.20, 2.00, 0.05),
        )
    else:
        raise ValueError("unsupported profile")
    values = np.concatenate([arange_inclusive(*segment) for segment in segments])
    return np.array(sorted({round(float(value), 12) for value in values}), dtype=float)


def spectrum_bands(
    *,
    spectrum_type: str,
    significant_wave_height: float,
    peak_period: float,
    gamma: float,
    omega_min: float,
    omega_max: float,
) -> dict[str, object]:
    dense = np.linspace(omega_min, omega_max, 10000)
    density = wave_spectrum_density(
        dense,
        spectrum_type=spectrum_type,
        significant_wave_height=significant_wave_height,
        peak_period=peak_period,
        gamma=gamma,
    )
    half = 0.5 * float(np.max(density))
    half_band = dense[density >= half]
    moment_density = density * np.gradient(dense)
    cumulative = np.cumsum(moment_density)
    cumulative /= cumulative[-1]

    def quantile(value: float) -> float:
        return float(dense[min(np.searchsorted(cumulative, value), dense.size - 1)])

    return {
        "dense_omega_peak": float(dense[int(np.argmax(density))]),
        "dense_spectrum_peak": float(np.max(density)),
        "half_power_band": [float(half_band[0]), float(half_band[-1])],
        "energy_band_05_95": [quantile(0.05), quantile(0.95)],
        "energy_band_01_99": [quantile(0.01), quantile(0.99)],
        "dense_m0": float(np.trapz(density, dense)),
    }


def grid_diagnostics(
    omega: np.ndarray,
    *,
    spectrum_type: str,
    significant_wave_height: float,
    peak_period: float,
    gamma: float,
) -> dict[str, object]:
    density = wave_spectrum_density(
        omega,
        spectrum_type=spectrum_type,
        significant_wave_height=significant_wave_height,
        peak_period=peak_period,
        gamma=gamma,
    )
    widths = spectral_component_widths(omega)
    amplitudes = np.sqrt(2.0 * density * widths)
    bands = spectrum_bands(
        spectrum_type=spectrum_type,
        significant_wave_height=significant_wave_height,
        peak_period=peak_period,
        gamma=gamma,
        omega_min=max(1.0e-4, float(omega[0])),
        omega_max=float(omega[-1]),
    )
    half_low, half_high = bands["half_power_band"]
    energy_low, energy_high = bands["energy_band_05_95"]
    half_mask = (omega >= half_low) & (omega <= half_high)
    energy_mask = (omega >= energy_low) & (omega <= energy_high)
    return {
        "omega_count": int(omega.size),
        "omega_min": float(omega[0]),
        "omega_max": float(omega[-1]),
        "delta_omega_min": float(np.min(np.diff(omega))),
        "delta_omega_max": float(np.max(np.diff(omega))),
        "delta_omega_mean": float(np.mean(np.diff(omega))),
        "omega_peak": float(2.0 * np.pi / peak_period),
        "spectrum_grid_peak_omega": float(omega[int(np.argmax(density))]),
        "spectrum_grid_peak_density": float(np.max(density)),
        "trapz_m0": float(np.trapz(density, omega)),
        "component_m0": float(0.5 * np.sum(amplitudes**2)),
        "component_hs": float(4.0 * np.sqrt(0.5 * np.sum(amplitudes**2))),
        "points_in_half_power_band": int(np.count_nonzero(half_mask)),
        "points_in_energy_05_95_band": int(np.count_nonzero(energy_mask)),
        "half_power_omega_values": [float(value) for value in omega[half_mask]],
        "energy_05_95_omega_values": [float(value) for value in omega[energy_mask]],
        "bands": bands,
    }


def plot_grid(path: Path, omega: np.ndarray, diagnostics: dict[str, object], *, args: argparse.Namespace) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    dense = np.linspace(args.omega_min, args.omega_max, 3000)
    dense_density = wave_spectrum_density(
        dense,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=args.peak_period,
        gamma=args.peak_enhancement_factor,
    )
    density = wave_spectrum_density(
        omega,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=args.peak_period,
        gamma=args.peak_enhancement_factor,
    )
    widths = spectral_component_widths(omega)
    amplitudes = np.sqrt(2.0 * density * widths)
    half_low, half_high = diagnostics["bands"]["half_power_band"]
    energy_low, energy_high = diagnostics["bands"]["energy_band_05_95"]
    omega_peak = 2.0 * np.pi / args.peak_period

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.4), sharex=True)
    axes[0].plot(dense, dense_density, color="#111111", linewidth=1.5, label="dense spectrum")
    axes[0].scatter(omega, density, color="#d62728", s=18, label="designed grid")
    axes[0].axvline(omega_peak, color="#2ca02c", linestyle="--", linewidth=1.0, label="omega_p")
    axes[0].axvspan(half_low, half_high, color="#ffcc66", alpha=0.25, label="half-power")
    axes[0].set_ylabel("S_eta(omega)")
    axes[0].set_title("Spectrum-focused frequency grid")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)

    axes[1].bar(omega, amplitudes, width=np.minimum(widths, 0.025), color="#1f77b4", alpha=0.75)
    axes[1].axvspan(energy_low, energy_high, color="#99ddff", alpha=0.22, label="5-95% energy")
    axes[1].axvline(omega_peak, color="#2ca02c", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("a_j (m)")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].legend(frameon=False)

    axes[2].plot(omega[:-1], np.diff(omega), marker="o", markersize=3, color="#9467bd", linewidth=1.0)
    axes[2].axvspan(half_low, half_high, color="#ffcc66", alpha=0.25)
    axes[2].set_xlabel("Angular frequency (rad/s)")
    axes[2].set_ylabel("Delta omega")
    axes[2].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_omega_files(output_root: Path, omega: np.ndarray) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    txt = output_root / "omega_values.txt"
    csv = output_root / "omega_values.csv"
    txt.write_text("\n".join(f"{value:.12g}" for value in omega) + "\n", encoding="utf-8")
    csv.write_text("index,omega_rad_s\n" + "\n".join(f"{index},{value:.12g}" for index, value in enumerate(omega)) + "\n", encoding="utf-8")
    return {"txt": txt, "csv": csv}


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    base = segmented_grid(args.profile)
    include = np.array(args.include_omega or [], dtype=float)
    omega = np.array(sorted({round(float(value), 12) for value in np.concatenate([base, include])}), dtype=float)
    omega = omega[(omega >= args.omega_min) & (omega <= args.omega_max)]
    diagnostics = grid_diagnostics(
        omega,
        spectrum_type=args.spectrum_type,
        significant_wave_height=args.significant_wave_height,
        peak_period=args.peak_period,
        gamma=args.peak_enhancement_factor,
    )
    files = write_omega_files(output_root, omega)
    figure = plot_grid(output_root / "figures" / "spectrum_focused_frequency_grid.png", omega, diagnostics, args=args)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "spectrum_type": args.spectrum_type,
        "significant_wave_height": args.significant_wave_height,
        "peak_period": args.peak_period,
        "peak_enhancement_factor": args.peak_enhancement_factor,
        "profile": args.profile,
        "diagnostics": diagnostics,
        "omega_files": files,
        "figure": figure,
        "hydrodynamic_generation_command": (
            ".\\.venv\\Scripts\\python.exe scripts\\generate_dm10_cummins_hydrodynamics.py "
            "--output data\\external\\DM-FEM2D\\HydrodynamicData\\Yoga\\DM10_direction0_cummins_spectrum_dense_mesh2.nc "
            f"--omega-file {files['txt']} --mesh-size 2.0 --vertical-mesh-size 2.0 --n-jobs 1 --force"
        ),
    }
    metrics_path = write_metrics_json(output_root / "spectrum_frequency_grid_metrics.json", metrics)
    print("Spectrum-focused frequency grid designed.")
    print(f"omega_count: {diagnostics['omega_count']}")
    print(f"points_in_half_power_band: {diagnostics['points_in_half_power_band']}")
    print(f"points_in_energy_05_95_band: {diagnostics['points_in_energy_05_95_band']}")
    print(f"omega_values: {files['txt']}")
    print(f"figure: {figure}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
