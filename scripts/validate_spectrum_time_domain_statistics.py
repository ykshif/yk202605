"""Validate spectrum-driven time-domain outputs with variance and RMS checks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_multi_harmonic_amplitudes,
    harmonic_component_variance,
    relative_l2_error,
    wave_spectrum_density,
    zero_mean_rms,
)


DEFAULT_CASE_ROOT = REPO_ROOT / "results" / "time_domain" / "spectrum_jonswap_dm10_mesh2_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--discard-seconds", type=float, default=None)
    parser.add_argument("--discard-peak-cycles", type=float, default=2.0)
    return parser.parse_args()


def load_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def representative_columns(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("heave array must contain at least three columns")
    return (0, count // 2, count - 1)


def windows_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def save_figure(fig, path: Path, *, dpi: int) -> Path:
    """Save Matplotlib figures robustly when Windows paths approach 260 chars."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32" and len(str(path.resolve())) >= 240:
        with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as stream:
            temporary_path = Path(stream.name)
        try:
            fig.savefig(temporary_path, dpi=dpi)
            with temporary_path.open("rb") as source, open(windows_long_path(path), "wb") as target:
                shutil.copyfileobj(source, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    fig.savefig(path, dpi=dpi)
    return path


def write_centerline_rms_csv(
    path: Path,
    time_rms: np.ndarray,
    harmonic_rms: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("centerline_index", "x_over_L", "time_rms", "harmonic_fit_rms", "relative_delta"))
        x = np.linspace(0.0, 1.0, time_rms.size)
        for index, (time_value, harmonic_value) in enumerate(zip(time_rms, harmonic_rms)):
            relative_delta = (harmonic_value - time_value) / max(abs(time_value), 1.0e-30)
            writer.writerow([index, float(x[index]), float(time_value), float(harmonic_value), float(relative_delta)])
    return path


def plot_wave_variance(path: Path, labels: list[str], values: list[float]) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("Variance (m^2)")
    ax.set_title("Wave-elevation variance closure")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, path, dpi=240)
    plt.close(fig)
    return path


def plot_wave_component_recovery(
    path: Path,
    omega: np.ndarray,
    input_amplitude: np.ndarray,
    fitted_amplitude: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(omega, input_amplitude, color="#111111", linewidth=1.4, label="input")
    ax.plot(omega, np.abs(fitted_amplitude), color="#d62728", linestyle="--", linewidth=1.2, label="fitted")
    ax.set_xlabel("Angular frequency (rad/s)")
    ax.set_ylabel("Wave component amplitude (m)")
    ax.set_title("Wave component recovery from synthesized time series")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, path, dpi=240)
    plt.close(fig)
    return path


def plot_centerline_rms(path: Path, time_rms: np.ndarray, harmonic_rms: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, time_rms.size)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(x, time_rms, color="#1f77b4", linewidth=1.5, label="time RMS")
    ax.plot(x, harmonic_rms, color="#d62728", linestyle="--", linewidth=1.3, label="component-fit RMS")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RMS")
    ax.set_title("Centerline heave RMS closure")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, path, dpi=240)
    plt.close(fig)
    return path


def plot_response_component_spectra(
    path: Path,
    omega: np.ndarray,
    fitted_heave: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    columns = representative_columns(fitted_heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for column, label in zip(columns, labels):
        ax.plot(omega, np.abs(fitted_heave[:, column]), linewidth=1.2, label=label)
    ax.set_xlabel("Angular frequency (rad/s)")
    ax.set_ylabel("Heave component amplitude")
    ax.set_title("Fitted response component amplitudes")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    case_root = args.case_root.resolve()
    output_root = (args.output_root or case_root / "statistics_validation").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((case_root / "metrics.json").read_text(encoding="utf-8"))
    time = load_array(case_root, "time.npy")
    omega = load_array(case_root, "wave_component_omega.npy")
    wave_amplitude = load_array(case_root, "wave_component_amplitude.npy")
    wave_phase = load_array(case_root, "wave_component_phase.npy")
    wave_elevation = load_array(case_root, "wave_elevation_time.npy")
    heave = load_array(case_root, "centerline_heave_time.npy")
    excitation_force = load_array(case_root, "excitation_force_time.npy")

    discard_seconds = (
        args.discard_seconds
        if args.discard_seconds is not None
        else args.discard_peak_cycles * float(metrics["peak_period_s"])
    )
    mask = time >= (time[0] + discard_seconds)
    if np.count_nonzero(mask) <= 2 * omega.size + 1:
        raise ValueError("not enough post-discard samples for multi-harmonic fitting")
    fit_start = float(time[mask][0])

    wave_fit = fit_multi_harmonic_amplitudes(wave_elevation, time, omega, start_time=fit_start)
    heave_fit = fit_multi_harmonic_amplitudes(heave, time, omega, start_time=fit_start)
    force_fit = fit_multi_harmonic_amplitudes(excitation_force, time, omega, start_time=fit_start)

    discrete_wave_variance = float(harmonic_component_variance(wave_amplitude))
    target_wave_complex_amplitude = wave_amplitude * np.exp(-1j * wave_phase)
    fitted_wave_variance = float(harmonic_component_variance(wave_fit))
    time_wave_variance = float(zero_mean_rms(wave_elevation[mask]) ** 2)
    target_spectrum = wave_spectrum_density(
        omega,
        spectrum_type=str(metrics["spectrum_type"]),
        significant_wave_height=float(metrics["significant_wave_height"]),
        peak_period=float(metrics["peak_period_s"]),
    )
    target_trapz_variance = float(np.trapz(target_spectrum, omega))

    heave_time_rms = zero_mean_rms(heave[mask], axis=0)
    heave_fit_rms = np.sqrt(harmonic_component_variance(heave_fit, axis=0))
    force_time_rms = zero_mean_rms(excitation_force[mask], axis=0)
    force_fit_rms = np.sqrt(harmonic_component_variance(force_fit, axis=0))

    figures = [
        plot_wave_variance(
            output_root / "figures" / "wave_variance_closure.png",
            ["target", "components", "time"],
            [target_trapz_variance, discrete_wave_variance, time_wave_variance],
        ),
        plot_wave_component_recovery(
            output_root / "figures" / "wave_component_recovery.png",
            omega,
            wave_amplitude,
            wave_fit,
        ),
        plot_centerline_rms(
            output_root / "figures" / "centerline_heave_rms_closure.png",
            heave_time_rms,
            heave_fit_rms,
        ),
        plot_response_component_spectra(
            output_root / "figures" / "representative_heave_component_spectra.png",
            omega,
            heave_fit,
        ),
    ]
    centerline_csv = write_centerline_rms_csv(
        output_root / "centerline_rms_validation.csv",
        heave_time_rms,
        heave_fit_rms,
    )

    validation = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_root": case_root,
        "discard_seconds": discard_seconds,
        "fit_start_time_s": fit_start,
        "component_count": int(omega.size),
        "target_trapz_wave_variance": target_trapz_variance,
        "discrete_component_wave_variance": discrete_wave_variance,
        "fitted_component_wave_variance": fitted_wave_variance,
        "time_wave_variance": time_wave_variance,
        "discrete_component_hs": float(4.0 * np.sqrt(discrete_wave_variance)),
        "time_series_hs": float(4.0 * np.sqrt(time_wave_variance)),
        "wave_component_amplitude_l2_relative_error": relative_l2_error(np.abs(wave_fit), wave_amplitude),
        "wave_component_complex_l2_relative_error": relative_l2_error(wave_fit, target_wave_complex_amplitude),
        "wave_variance_time_vs_component_relative_error": abs(time_wave_variance - discrete_wave_variance)
        / max(discrete_wave_variance, 1.0e-30),
        "wave_variance_fit_vs_component_relative_error": abs(fitted_wave_variance - discrete_wave_variance)
        / max(discrete_wave_variance, 1.0e-30),
        "centerline_heave_rms_l2_relative_error": relative_l2_error(heave_fit_rms, heave_time_rms),
        "centerline_heave_time_rms_max": float(np.max(heave_time_rms)),
        "centerline_heave_fit_rms_max": float(np.max(heave_fit_rms)),
        "excitation_force_rms_l2_relative_error": relative_l2_error(force_fit_rms, force_time_rms),
        "centerline_csv": centerline_csv,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "statistics_metrics.json", validation)

    print("Spectrum time-domain statistics validation completed.")
    print(f"wave_variance_time_vs_component_relative_error: {validation['wave_variance_time_vs_component_relative_error']:.6g}")
    print(f"centerline_heave_rms_l2_relative_error: {validation['centerline_heave_rms_l2_relative_error']:.6g}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
