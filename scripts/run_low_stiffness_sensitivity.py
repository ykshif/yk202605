"""Evaluate very-low released stiffness levels for the 10x10 hinge case.

The main boundary18 DOE already includes ``0``, ``1e6`` and ``1e7``.  This
script fills the missing lower decade around ``1e5`` and compares whether
very-low elastic hinges are still distinguishable from the ideal hinged limit.

It is intentionally small and single-frequency: the goal is to support the
paper discussion around whether the lower stiffness bound matters before
launching a larger 18D/180D optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    BoundaryStiffnessDesign,
    MetricObjective,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    evaluate_design_response,
    mark_pareto_rows,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
    solve_complex_hinge_case,
)

from run_boundary18_fullrange_single_frequency import (  # noqa: E402
    summarize_connector_relative_motion,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "low_stiffness_sensitivity"


def _safe_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{float(value):.2e}".replace("+", "").replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def low_stiffness_samples() -> list[dict[str, Any]]:
    """Return uniform and orientation-asymmetric very-low-stiffness designs."""

    samples: list[dict[str, Any]] = []
    uniform_values = (0.0, 1.0e4, 1.0e5, 1.0e6, 3.162277660168379e6, 1.0e7, 3.162277660168379e7, 1.0e8)
    for value in uniform_values:
        samples.append(
            {
                "design_label": f"uniform_{_safe_label(value)}",
                "family": "uniform",
                "values": tuple([value] * 18),
                "description": "All 18 complete internal boundaries use the same released stiffness.",
            }
        )

    orientation_pairs = (
        (1.0e6, 1.0e5),
        (1.0e5, 1.0e6),
        (1.0e7, 1.0e6),
        (1.0e6, 1.0e7),
        (1.0e7, 0.0),
        (0.0, 1.0e7),
    )
    for x_value, y_value in orientation_pairs:
        samples.append(
            {
                "design_label": f"orient_x_{_safe_label(x_value)}_y_{_safe_label(y_value)}",
                "family": "orientation_low",
                "values": tuple([x_value] * 9 + [y_value] * 9),
                "description": "All x boundaries and all y boundaries use different very-low stiffness levels.",
            }
        )
    return samples


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _display_label(label: str) -> str:
    replacements = {
        "uniform_0": "hinged\n0",
        "uniform_1p00e04": "$10^4$",
        "uniform_1p00e05": "$10^5$",
        "uniform_1p00e06": "$10^6$",
        "uniform_3p16e06": "$3.16\\times10^6$",
        "uniform_1p00e07": "$10^7$",
        "uniform_3p16e07": "$3.16\\times10^7$",
        "uniform_1p00e08": "$10^8$",
        "orient_x_1p00e06_y_1p00e05": "$x:10^6$\n$y:10^5$",
        "orient_x_1p00e05_y_1p00e06": "$x:10^5$\n$y:10^6$",
        "orient_x_1p00e07_y_1p00e06": "$x:10^7$\n$y:10^6$",
        "orient_x_1p00e06_y_1p00e07": "$x:10^6$\n$y:10^7$",
        "orient_x_1p00e07_y_0": "$x:10^7$\n$y:0$",
        "orient_x_0_y_1p00e07": "$x:0$\n$y:10^7$",
    }
    return replacements.get(label, label.replace("_", "\n"))


def _plot_results(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    plt = _ensure_matplotlib()
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    uniform_rows = [row for row in rows if row["family"] == "uniform"]
    labels = [_display_label(str(row["design_label"])) for row in uniform_rows]
    x = np.arange(len(uniform_rows))
    metrics = [
        ("mean_heave", "mean heave (m)", "#2f9e44", 1.0),
        ("max_released_relative_rotation_envelope", "max released relative rotation (rad)", "#5f3dc4", 1.0),
        ("max_connector_bending_envelope", "max connector bending envelope ($\\times10^6$)", "#e67700", 1.0e-6),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    for ax, (key, ylabel, color, scale) in zip(axes, metrics):
        values = np.array([float(row[key]) * scale for row in uniform_rows])
        ax.plot(x, values, marker="o", color=color, linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
    fig.suptitle("Uniform very-low released stiffness sensitivity")
    path = figure_root / "low_stiffness_uniform_metrics.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=260)
    paths.append(path)
    plt.close(fig)

    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    rotation = np.array([float(row["max_released_relative_rotation_envelope"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows]) / 1.0e6
    is_pareto = np.array([str(row["is_pareto"]).lower() == "true" for row in rows])
    family = np.array([str(row["family"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    ax.scatter(mean_heave[family == "uniform"], rotation[family == "uniform"], color="#f08c00", s=78, label="uniform")
    ax.scatter(
        mean_heave[family != "uniform"],
        rotation[family != "uniform"],
        color="#1c7ed6",
        marker="D",
        s=58,
        label="orientation-asymmetric",
    )
    ax.scatter(
        mean_heave[is_pareto],
        rotation[is_pareto],
        facecolors="none",
        edgecolors="#212529",
        s=135,
        linewidths=1.1,
        label="Pareto in low-stiffness set",
    )
    for row, bend in zip(rows, bending):
        if row["design_label"] in {"uniform_0", "uniform_1p00e05", "uniform_1p00e06", "uniform_1p00e07"}:
            ax.annotate(
                _display_label(str(row["design_label"])),
                (float(row["mean_heave"]), float(row["max_released_relative_rotation_envelope"])),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max released relative rotation (rad)")
    ax.set_title("Low-stiffness designs: motion vs relative rotation")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    path = figure_root / "low_stiffness_pareto_projection.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=260)
    paths.append(path)
    plt.close(fig)
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    response_root = output_root / "responses"
    response_root.mkdir(parents=True, exist_ok=True)

    base_case = build_complex_hinge_10x10_case(
        args.data_root,
        k_hinge=args.coupling_stiffness,
    )
    groups = build_hinge_design_groups(base_case, "continuous_boundary")
    group_names = [group.name for group in groups]

    rows: list[dict[str, Any]] = []
    samples = low_stiffness_samples()
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] solving {sample['design_label']}", flush=True)
        design = BoundaryStiffnessDesign(
            values=sample["values"],
            grouping="continuous_boundary",
            parameter="released_dof_stiffness",
            coupling_stiffness=args.coupling_stiffness,
            label=sample["design_label"],
            meta={"family": sample["family"], "sample_description": sample["description"]},
        )
        value_by_group = design.values_for_groups(group_names)
        case = apply_grouped_hinge_stiffness(
            base_case,
            groups,
            value_by_group,
            parameter="released_dof_stiffness",
        )
        start = time.perf_counter()
        solved = solve_complex_hinge_case(case)
        elapsed = time.perf_counter() - start
        np.save(response_root / f"response_{sample['design_label']}.npy", solved.response)
        np.save(response_root / f"heave_grid_{sample['design_label']}.npy", solved.heave_grid_merged)

        evaluation = evaluate_design_response(
            case,
            solved.response,
            solved.omega,
            design=design.as_dict(group_names),
            scenario={
                "omega": solved.omega,
                "frequency_index": args.frequency_index,
                "wave_direction_deg": 0.0,
                "scenario_label": "low_stiffness_sensitivity",
            },
            heave_grid=solved.heave_grid_merged,
            cid_prefix=sample["design_label"],
        )
        row = evaluation.summary_row()
        row.update(
            summarize_connector_relative_motion(
                case,
                solved.response,
                solved.omega,
                cid_prefix=f"{sample['design_label']}_delta",
            )
        )
        row["sample_index"] = index
        row["solve_elapsed_s"] = elapsed
        rows.append(row)

    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective("released_rotation", "max_released_relative_rotation_envelope", minimize=True),
        MetricObjective("connector_bending", "max_connector_bending_envelope", minimize=True),
    )
    rows = mark_pareto_rows(rows, objectives)
    summary_path = output_root / "low_stiffness_sensitivity_summary.csv"
    _write_csv(summary_path, rows)
    figure_paths = _plot_results(rows, output_root)
    manifest = {
        "sample_count": len(rows),
        "pareto_count": sum(1 for row in rows if row["is_pareto"]),
        "summary_path": str(summary_path),
        "figures": [str(path) for path in figure_paths],
    }
    (output_root / "low_stiffness_sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run very-low stiffness sensitivity for the 10x10 hinge case.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--coupling-stiffness", type=float, default=1.0e10)
    parser.add_argument("--frequency-index", type=int, default=0)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
