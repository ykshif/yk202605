"""Run, validate, and plot the 300 m RODM reference-case workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import (  # noqa: E402
    build_workflow_paths,
    build_rodm_frequency_case_from_config,
    load_case_config,
    write_metrics_json,
)
from offshore_energy_sim.postprocess.metrics import rmse  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    extract_centerline_heave,
    load_xy,
)
from offshore_energy_sim.postprocess.validation import (  # noqa: E402
    curve_error_metrics,
    interpolated_curve_rmse,
    response_error_metrics,
)
from offshore_energy_sim.postprocess.workflow_report import write_workflow_report  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete 300 m reference-case workflow.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "reference_case_300.yaml",
        help="Path to the reference-case YAML file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for relative baseline/output paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Case output directory. Defaults to results/<case_id>.",
    )
    return parser.parse_args()


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _solve_case(config_path: Path):
    case = build_rodm_frequency_case_from_config(config_path)
    start = time.perf_counter()
    result = solve_rodm_frequency_case(case)
    elapsed = time.perf_counter() - start
    return case, result.global_displacement, elapsed


def _heave_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    x_candidate, heave_candidate = extract_centerline_heave(candidate)
    x_baseline, heave_baseline = extract_centerline_heave(baseline)
    return curve_error_metrics(
        x_candidate,
        heave_candidate,
        x_baseline,
        heave_baseline,
        quantity_prefix="heave",
    )


def _external_curve_metrics(candidate: np.ndarray, exp_file: Path, fu_file: Path) -> dict[str, float]:
    x_candidate, heave_candidate = extract_centerline_heave(candidate)
    exp_x, exp_y = load_xy(exp_file)
    fu_x, fu_y = load_xy(fu_file)
    return {
        "rmse_vs_exp300": interpolated_curve_rmse(x_candidate, heave_candidate, exp_x, exp_y),
        "rmse_vs_fu_sim300": interpolated_curve_rmse(x_candidate, heave_candidate, fu_x, fu_y),
    }


def _write_solver_plot(
    figure_dir: Path,
    saved_response: np.ndarray,
    default_response: np.ndarray,
    exp_file: Path,
    fu_file: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    x_saved, heave_saved = extract_centerline_heave(saved_response)
    x_default, heave_default = extract_centerline_heave(default_response)
    x_exp, heave_exp = load_xy(exp_file)
    x_fu, heave_fu = load_xy(fu_file)

    default_rmse = rmse(heave_default, heave_saved)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "reference_case_300_workflow_comparison.png"
    pdf_path = figure_dir / "reference_case_300_workflow_comparison.pdf"

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_saved, heave_saved, color="#111111", linewidth=2.2, label="Saved reference response")
    ax.plot(
        x_default,
        heave_default,
        color="#1f77b4",
        linewidth=1.7,
        linestyle="--",
        label=f"Config RODM / DM_Method (RMSE={default_rmse:.4f})",
    )
    ax.scatter(
        x_exp,
        heave_exp,
        color="#d62728",
        s=30,
        marker="o",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label="Experiment",
    )
    ax.plot(x_fu, heave_fu, color="#666666", linewidth=1.3, linestyle=":", label="Fu et al. simulation")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title("300 m x 60 m Floating Body, Config Workflow")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.4)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _write_reference_workflow_report(
    report_path: Path,
    *,
    config_path: Path,
    default_elapsed: float,
    default_response_path: Path,
    figure_png: Path,
    figure_pdf: Path,
    metrics: dict[str, dict[str, object]],
) -> None:
    write_workflow_report(
        report_path,
        title="Reference Case 300 Workflow Report",
        scope_lines=[
            "This report was generated by the config-driven reference-case workflow.",
            "Numerical-result expectation: unchanged relative to the validated packaged",
            "solver output for the default RODM / DM_Method-equivalent case.",
        ],
        input_output_lines=[
            f"- config: `{config_path}`",
            f"- default_elapsed_seconds: `{default_elapsed:.3f}`",
            f"- default_response: `{default_response_path}`",
            f"- figure_png: `{figure_png}`",
            f"- figure_pdf: `{figure_pdf}`",
        ],
        metric_sections=metrics,
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    config = load_case_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]

    baseline_response_path = _resolve_path(repo_root, outputs["baseline_displacement"])
    exp_file = _resolve_path(repo_root, inputs["experiment_heave_rao"])
    fu_file = _resolve_path(repo_root, inputs["fu_2007_heave_rao"])
    saved_response = np.load(baseline_response_path)

    default_case, default_response, default_elapsed = _solve_case(args.config)

    case_root = args.output_dir or (repo_root / "results" / default_case.case_id)
    default_paths = build_workflow_paths(case_root, variant_id="default")

    np.save(default_paths.response_path, default_response)

    figure_png, figure_pdf = _write_solver_plot(
        default_paths.figures_dir,
        saved_response,
        default_response,
        exp_file,
        fu_file,
    )

    metrics = {
        "Default Response vs Saved Baseline": response_error_metrics(default_response, saved_response),
        "Default Heave vs Saved Baseline": _heave_metrics(default_response, saved_response),
        "Default Heave vs External Curves": _external_curve_metrics(default_response, exp_file, fu_file),
    }

    default_variant_metrics = {
        "response_vs_saved_baseline": metrics["Default Response vs Saved Baseline"],
        "heave_vs_saved_baseline": metrics["Default Heave vs Saved Baseline"],
        "heave_vs_external_curves": metrics["Default Heave vs External Curves"],
        "elapsed_seconds": default_elapsed,
        "response_path": default_paths.response_path,
    }
    write_metrics_json(default_paths.metrics_path, default_variant_metrics)
    write_metrics_json(
        default_paths.case_root / "metrics.json",
        {
            "case_id": default_case.case_id,
            "default": default_variant_metrics,
        },
    )

    report_path = default_paths.report_path
    _write_reference_workflow_report(
        report_path,
        config_path=args.config,
        default_elapsed=default_elapsed,
        default_response_path=default_paths.response_path,
        figure_png=figure_png,
        figure_pdf=figure_pdf,
        metrics=metrics,
    )

    print(f"case_id: {default_case.case_id}")
    print(f"default_response: {default_paths.response_path}")
    print(f"figure_png: {figure_png}")
    print(f"figure_pdf: {figure_pdf}")
    print(f"report: {report_path}")
    print("Reference case 300 workflow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
