"""Run the refactored RODM solver and compare against the 300 m baseline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    build_rodm_frequency_case,
    default_paths,
    extract_centerline_heave,
    load_xy,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


RESULT_DIR = REPO_ROOT / "results"
REPORT_PATH = REPO_ROOT / "docs" / "rodm_full_regression_report.md"
GENERATED_RESPONSE_PATH = RESULT_DIR / "reference_case_300_rodm_generated.npy"
GENERATED_REVERSED_RESPONSE_PATH = RESULT_DIR / "reference_case_300_rodm_hydro_reversed.npy"


def rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def compare_response(generated: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    difference = generated - baseline
    baseline_norm = np.linalg.norm(baseline)
    diff_norm = np.linalg.norm(difference)
    return {
        "shape_generated": tuple(generated.shape),
        "shape_baseline": tuple(baseline.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "l2_abs_error": float(diff_norm),
        "l2_relative_error": float(diff_norm / baseline_norm) if baseline_norm else float("nan"),
    }


def compare_heave(generated: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    x_generated, heave_generated = extract_centerline_heave(generated)
    x_baseline, heave_baseline = extract_centerline_heave(baseline)
    if not np.allclose(x_generated, x_baseline):
        raise AssertionError("Generated and baseline heave x/L grids differ.")
    difference = heave_generated - heave_baseline
    return {
        "heave_len": int(heave_generated.size),
        "heave_max_abs_error": float(np.max(np.abs(difference))),
        "heave_rmse": rmse(heave_generated, heave_baseline),
        "heave_generated_min": float(np.min(heave_generated)),
        "heave_generated_max": float(np.max(heave_generated)),
        "heave_generated_mean": float(np.mean(heave_generated)),
        "heave_baseline_min": float(np.min(heave_baseline)),
        "heave_baseline_max": float(np.max(heave_baseline)),
        "heave_baseline_mean": float(np.mean(heave_baseline)),
    }


def generated_experiment_metrics(generated: np.ndarray) -> dict[str, float]:
    paths = default_paths(REPO_ROOT)
    x_generated, heave_generated = extract_centerline_heave(generated)
    exp_x, exp_y = load_xy(paths.experiment_file)
    fu_x, fu_y = load_xy(paths.fu_sim_file)
    return {
        "generated_rmse_vs_exp300": rmse(np.interp(exp_x, x_generated, heave_generated), exp_y),
        "generated_rmse_vs_fu_sim300": rmse(np.interp(fu_x, x_generated, heave_generated), fu_y),
    }


def format_metric_table(metrics: dict[str, object]) -> list[str]:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | `{value}` |")
    return lines


def format_report(
    elapsed_seconds: float,
    response_metrics: dict[str, object],
    heave_metrics: dict[str, object],
    experiment_metrics: dict[str, float],
    reversed_elapsed_seconds: float,
    reversed_response_metrics: dict[str, object],
    reversed_heave_metrics: dict[str, object],
    reversed_experiment_metrics: dict[str, float],
) -> str:
    lines = [
        "# RODM Full Regression Report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        "This report compares the refactored `solve_rodm_frequency_case()` output",
        "against the existing `displacement_55mesh_300.npy` baseline for the",
        "300 m x 60 m floating-body reference case.",
        "",
        "Expected numerical-result change: none. Any nonzero difference should be",
        "investigated before legacy scripts are redirected to the package solver.",
        "",
        "## Runtime",
        "",
        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
        f"- generated_response: `{GENERATED_RESPONSE_PATH.relative_to(REPO_ROOT)}`",
        f"- hydro_reversed_elapsed_seconds: `{reversed_elapsed_seconds:.3f}`",
        f"- hydro_reversed_response: `{GENERATED_REVERSED_RESPONSE_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Default Solver vs Baseline",
        "",
    ]
    lines.extend(format_metric_table(response_metrics))

    lines.extend(
        [
            "",
            "## Default Solver Centerline Heave",
            "",
        ]
    )
    lines.extend(format_metric_table(heave_metrics))

    lines.extend(
        [
            "",
            "## Default Solver vs External Curves",
            "",
        ]
    )
    lines.extend(format_metric_table(experiment_metrics))

    lines.extend(
        [
            "",
            "## Hydrodynamic-Node-Reversed Candidate vs Baseline",
            "",
            "This candidate reverses the 10 hydrodynamic node blocks before solving.",
            "It is not the default legacy-equivalent path, but it matches the saved",
            "baseline heave curve much more closely and may reflect the historical",
            "notebook path that created `displacement_55mesh_300.npy`.",
            "",
        ]
    )
    lines.extend(format_metric_table(reversed_response_metrics))

    lines.extend(
        [
            "",
            "## Hydrodynamic-Node-Reversed Centerline Heave",
            "",
        ]
    )
    lines.extend(format_metric_table(reversed_heave_metrics))

    lines.extend(
        [
            "",
            "## Hydrodynamic-Node-Reversed vs External Curves",
            "",
        ]
    )
    lines.extend(format_metric_table(reversed_experiment_metrics))

    return "\n".join(lines) + "\n"


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)

    paths = default_paths(REPO_ROOT)
    case = build_rodm_frequency_case(paths)

    start = time.perf_counter()
    result = solve_rodm_frequency_case(case)
    elapsed = time.perf_counter() - start

    generated = result.global_displacement
    baseline = np.load(paths.response_file)

    np.save(GENERATED_RESPONSE_PATH, generated)

    response_metrics = compare_response(generated, baseline)
    heave_metrics = compare_heave(generated, baseline)
    experiment_metrics = generated_experiment_metrics(generated)

    reversed_case = build_rodm_frequency_case(paths, reverse_hydrodynamic_node_order=True)
    start = time.perf_counter()
    reversed_result = solve_rodm_frequency_case(reversed_case)
    reversed_elapsed = time.perf_counter() - start
    reversed_generated = reversed_result.global_displacement
    np.save(GENERATED_REVERSED_RESPONSE_PATH, reversed_generated)

    reversed_response_metrics = compare_response(reversed_generated, baseline)
    reversed_heave_metrics = compare_heave(reversed_generated, baseline)
    reversed_experiment_metrics = generated_experiment_metrics(reversed_generated)

    report = format_report(
        elapsed,
        response_metrics,
        heave_metrics,
        experiment_metrics,
        reversed_elapsed,
        reversed_response_metrics,
        reversed_heave_metrics,
        reversed_experiment_metrics,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    if response_metrics["shape_generated"] != response_metrics["shape_baseline"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
