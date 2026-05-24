"""Build a unified basic validation benchmark for the RODM time-domain layer."""

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
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    fit_harmonic_amplitude,
    harmonic_amplitude_error,
    harmonic_force_time_series,
    solve_linear_time_domain,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "basic_benchmark_validation"
DEFAULT_CUMMINS_METRICS = (
    REPO_ROOT
    / "results"
    / "time_domain"
    / "cummins_dm10_generated_mesh2_42freq_no_window_residual_trapezoidal"
    / "metrics.json"
)
DEFAULT_EXTRAPOLATION_METRICS = (
    REPO_ROOT
    / "results"
    / "time_domain"
    / "hydrodynamic_extrapolation_dm10_mesh2"
    / "hydrodynamic_extrapolation_metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cummins-metrics", type=Path, default=DEFAULT_CUMMINS_METRICS)
    parser.add_argument("--extrapolation-metrics", type=Path, default=DEFAULT_EXTRAPOLATION_METRICS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, object], keys: tuple[str, ...], default=None):
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def run_sdof_validation(output_root: Path) -> dict[str, object]:
    """Validate the generic Newmark solver against a frequency-domain SDOF result."""

    mass = np.array([[2.5]])
    damping = np.array([[0.7]])
    stiffness = np.array([[18.0]])
    force_hat = np.array([[3.0 + 1.5j]])
    omega = 1.35
    period = 2.0 * np.pi / omega
    time_step = period / 240.0
    time = np.arange(0.0, 80.0 * period + 0.5 * time_step, time_step)
    force = harmonic_force_time_series(force_hat.reshape(-1), omega, time)
    solved = solve_linear_time_domain(mass, damping, stiffness, force, time)
    reference = solve_frequency_domain(mass, damping, stiffness, force_hat, omega).reshape(-1)
    fitted = fit_harmonic_amplitude(
        solved.displacement,
        solved.time,
        omega,
        start_time=55.0 * period,
    )
    error = harmonic_amplitude_error(fitted, reference)
    figure = plot_sdof_validation(
        output_root / "figures" / "sdof_frequency_vs_time_validation.png",
        time,
        solved.displacement[:, 0],
        reference[0],
        omega,
        period,
    )
    return {
        "omega_rad_s": omega,
        "period_s": period,
        "time_step_s": time_step,
        "time_samples": int(time.size),
        "reference_complex_amplitude": complex_summary(reference[0]),
        "fitted_complex_amplitude": complex_summary(fitted[0]),
        "amplitude_error": error,
        "figure": figure,
    }


def complex_summary(value: complex) -> dict[str, float]:
    """Return a JSON-friendly complex-amplitude summary."""

    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "abs": float(abs(number)),
        "phase_rad": float(np.angle(number)),
    }


def plot_sdof_validation(
    path: Path,
    time: np.ndarray,
    displacement: np.ndarray,
    reference_amplitude: complex,
    omega: float,
    period: float,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = time >= time[-1] - 5.0 * period
    reference_signal = np.real(reference_amplitude * np.exp(-1j * omega * time[mask]))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(time[mask], displacement[mask], color="#1f77b4", linewidth=1.4, label="time-domain")
    ax.plot(time[mask], reference_signal, color="#d62728", linestyle="--", linewidth=1.2, label="frequency reference")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.set_title("SDOF Newmark validation")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def validation_rows(
    sdof: dict[str, object],
    cummins: dict[str, object],
    extrapolation: dict[str, object],
) -> list[dict[str, object]]:
    """Collect key validation errors into a flat table."""

    return [
        {
            "name": "SDOF Newmark",
            "value": nested(sdof, ("amplitude_error", "l2_relative_error")),
            "target": 2.0e-3,
            "kind": "deterministic",
        },
        {
            "name": "Constant A/B vs FD",
            "value": nested(cummins, ("constant_global_amplitude_error", "l2_relative_error")),
            "target": 1.0e-2,
            "kind": "regular_wave",
        },
        {
            "name": "Cummins vs FD",
            "value": nested(cummins, ("cummins_global_amplitude_error", "l2_relative_error")),
            "target": 1.0e-2,
            "kind": "regular_wave",
        },
        {
            "name": "Corrected B reconstruction",
            "value": nested(cummins, ("corrected_damping_relative_error_at_selected_omega",)),
            "target": 1.0e-12,
            "kind": "hydrodynamic_reconstruction",
        },
        {
            "name": "Original wave variance",
            "value": nested(
                extrapolation,
                (
                    "time_domain_comparison",
                    "comparisons",
                    "wave_variance_closure_error",
                    "before",
                ),
            ),
            "target": 3.0e-2,
            "kind": "spectrum_statistics",
        },
        {
            "name": "Extrapolated wave variance",
            "value": nested(
                extrapolation,
                (
                    "time_domain_comparison",
                    "comparisons",
                    "wave_variance_closure_error",
                    "after",
                ),
            ),
            "target": 1.0e-2,
            "kind": "spectrum_statistics",
        },
        {
            "name": "Original heave RMS",
            "value": nested(
                extrapolation,
                (
                    "time_domain_comparison",
                    "comparisons",
                    "centerline_heave_rms_closure_error",
                    "before",
                ),
            ),
            "target": 3.0e-2,
            "kind": "response_statistics",
        },
        {
            "name": "Extrapolated heave RMS",
            "value": nested(
                extrapolation,
                (
                    "time_domain_comparison",
                    "comparisons",
                    "centerline_heave_rms_closure_error",
                    "after",
                ),
            ),
            "target": 1.0e-2,
            "kind": "response_statistics",
        },
        {
            "name": "Kernel tail after/before",
            "value": nested(
                extrapolation,
                ("radiation_kernel_comparison", "tail_rms_ratio_after_over_before"),
            ),
            "target": 1.0,
            "kind": "kernel_stability",
        },
    ]


def plot_validation_summary(path: Path, rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["name"]) for row in rows]
    values = np.array([float(row["value"]) for row in rows], dtype=float)
    targets = np.array([float(row["target"]) for row in rows], dtype=float)
    colors = ["#2ca02c" if value <= target else "#d62728" for value, target in zip(values, targets)]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.barh(y, values, color=colors, alpha=0.85, label="measured")
    ax.scatter(targets, y, color="#111111", s=24, zorder=3, label="target")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("Relative error or diagnostic ratio")
    ax.set_title("Basic time-domain validation matrix")
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_spectrum_closure_comparison(
    path: Path,
    extrapolation: dict[str, object],
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["wave variance", "excitation RMS", "heave RMS"]
    before = np.array(
        [
            nested(extrapolation, ("time_domain_comparison", "comparisons", "wave_variance_closure_error", "before")),
            nested(extrapolation, ("time_domain_comparison", "comparisons", "excitation_rms_closure_error", "before")),
            nested(extrapolation, ("time_domain_comparison", "comparisons", "centerline_heave_rms_closure_error", "before")),
        ],
        dtype=float,
    )
    after = np.array(
        [
            nested(extrapolation, ("time_domain_comparison", "comparisons", "wave_variance_closure_error", "after")),
            nested(extrapolation, ("time_domain_comparison", "comparisons", "excitation_rms_closure_error", "after")),
            nested(extrapolation, ("time_domain_comparison", "comparisons", "centerline_heave_rms_closure_error", "after")),
        ],
        dtype=float,
    )
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2.0, before * 100.0, width, color="#1f77b4", label="original")
    ax.bar(x + width / 2.0, after * 100.0, width, color="#d62728", label="extrapolated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Closure error (%)")
    ax.set_title("JONSWAP-DM10 mesh2 statistics closure")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_benchmark_dashboard(
    path: Path,
    image_paths: list[Path],
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [item for item in image_paths if item.exists()]
    if len(existing) < 4:
        raise FileNotFoundError("at least four existing images are required for dashboard")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    titles = [
        "SDOF validation",
        "Validation matrix",
        "Cummins regular-wave heave",
        "Radiation kernel after extrapolation",
    ]
    for ax, image_path, title in zip(axes.ravel(), existing[:4], titles):
        ax.imshow(plt.imread(image_path))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sdof = run_sdof_validation(output_root)
    cummins = load_json(args.cummins_metrics)
    extrapolation = load_json(args.extrapolation_metrics)
    rows = validation_rows(sdof, cummins, extrapolation)
    failed = [row for row in rows if float(row["value"]) > float(row["target"])]

    summary_plot = plot_validation_summary(output_root / "figures" / "validation_error_summary.png", rows)
    spectrum_plot = plot_spectrum_closure_comparison(
        output_root / "figures" / "spectrum_closure_before_after.png",
        extrapolation,
    )
    cummins_heave_plot = (
        REPO_ROOT
        / "results"
        / "time_domain"
        / "cummins_dm10_generated_mesh2_42freq_no_window_residual_trapezoidal"
        / "figures"
        / "frequency_constant_cummins_heave_amplitude.png"
    )
    kernel_plot = (
        REPO_ROOT
        / "results"
        / "time_domain"
        / "hydrodynamic_extrapolation_dm10_mesh2"
        / "figures"
        / "radiation_kernel_after_extrapolation.png"
    )
    dashboard = plot_benchmark_dashboard(
        output_root / "figures" / "basic_benchmark_dashboard.png",
        [
            Path(sdof["figure"]),
            summary_plot,
            cummins_heave_plot,
            kernel_plot,
        ],
    )

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "passed" if not failed else "failed",
        "architecture_boundary": {
            "rodm_frequency_core_modified": False,
            "time_domain_adapter_external": True,
        },
        "sdof_validation": sdof,
        "cummins_metrics_path": args.cummins_metrics,
        "extrapolation_metrics_path": args.extrapolation_metrics,
        "validation_rows": rows,
        "failed_rows": failed,
        "figures": {
            "sdof": sdof["figure"],
            "summary": summary_plot,
            "spectrum_closure": spectrum_plot,
            "dashboard": dashboard,
            "cummins_heave_reference": cummins_heave_plot,
            "radiation_kernel_after_extrapolation": kernel_plot,
        },
    }
    metrics_path = write_metrics_json(output_root / "basic_benchmark_metrics.json", metrics)
    print("Basic time-domain benchmark validation completed.")
    print(f"status: {metrics['status']}")
    print(f"sdof_l2_relative_error: {nested(sdof, ('amplitude_error', 'l2_relative_error')):.6g}")
    print(f"cummins_l2_relative_error: {nested(cummins, ('cummins_global_amplitude_error', 'l2_relative_error')):.6g}")
    print(
        "extrapolated_heave_rms_closure_error: "
        f"{nested(extrapolation, ('time_domain_comparison', 'comparisons', 'centerline_heave_rms_closure_error', 'after')):.6g}"
    )
    print(f"dashboard: {dashboard}")
    print(f"metrics: {metrics_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
