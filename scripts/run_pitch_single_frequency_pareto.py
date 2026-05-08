"""Mark Pareto candidates for the fixed-frequency pitch-stiffness scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    MetricConstraint,
    MetricObjective,
    mark_pareto_rows,
)


DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "single_frequency_pitch_design_evaluation"
    / "pitch_single_frequency_evaluation_summary.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "single_frequency_pitch_design_pareto"

NUMERIC_FIELDS = {
    "pitch_stiffness",
    "coupling_stiffness",
    "omega",
    "frequency_index",
    "wave_direction_deg",
    "min_heave",
    "max_heave",
    "mean_heave",
    "connector_count",
    "max_connector_shear_envelope",
    "max_connector_bending_envelope",
    "max_released_moment_envelope",
}


def _load_summary_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in NUMERIC_FIELDS:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_constraints(args: argparse.Namespace) -> tuple[MetricConstraint, ...]:
    constraints: list[MetricConstraint] = []
    if args.max_shear is not None:
        constraints.append(
            MetricConstraint(
                "shear_limit",
                "max_connector_shear_envelope",
                upper_bound=args.max_shear,
            )
        )
    if args.max_bending is not None:
        constraints.append(
            MetricConstraint(
                "bending_limit",
                "max_connector_bending_envelope",
                upper_bound=args.max_bending,
            )
        )
    if args.max_released_moment is not None:
        constraints.append(
            MetricConstraint(
                "released_moment_limit",
                "max_released_moment_envelope",
                upper_bound=args.max_released_moment,
            )
        )
    return tuple(constraints)


def _plot_pareto(rows: list[dict], output_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows])
    feasible = np.array([row["is_feasible"] in (True, "True", "true", "1") for row in rows])
    pareto = np.array([row["is_pareto"] in (True, "True", "true", "1") for row in rows])
    labels = [str(row["pitch_stiffness_label"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    if np.any(feasible & ~pareto):
        ax.scatter(
            mean_heave[feasible & ~pareto],
            bending[feasible & ~pareto],
            s=54,
            color="#868e96",
            label="feasible dominated",
        )
    if np.any(~feasible):
        ax.scatter(
            mean_heave[~feasible],
            bending[~feasible],
            s=54,
            marker="x",
            color="#c92a2a",
            label="infeasible",
        )
    if np.any(pareto):
        ax.scatter(
            mean_heave[pareto],
            bending[pareto],
            s=76,
            color="#1c7ed6",
            label="Pareto candidate",
        )
        order = np.argsort(mean_heave[pareto])
        ax.plot(mean_heave[pareto][order], bending[pareto][order], color="#1c7ed6", linewidth=1.2)

    for label, x_value, y_value in zip(labels, mean_heave, bending):
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 5))

    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max connector bending envelope")
    ax.set_title("fixed-frequency pitch-stiffness Pareto screening")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "pitch_single_frequency_pareto_mean_heave_bending.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def run_pareto(args: argparse.Namespace) -> dict[str, Path | int]:
    summary_path = Path(args.summary).resolve()
    output_root = Path(args.output_root).resolve()
    rows = _load_summary_rows(summary_path)
    rows.sort(key=lambda row: float(row["pitch_stiffness"]))

    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective(
            "connector_bending",
            "max_connector_bending_envelope",
            minimize=True,
        ),
    )
    constraints = _build_constraints(args)
    marked = mark_pareto_rows(rows, objectives, constraints)
    for row in marked:
        row["objective_1"] = "minimize_mean_heave"
        row["objective_2"] = "minimize_max_connector_bending_envelope"
        row["constraint_mode"] = "user_bounds" if constraints else "unconstrained"

    output_path = output_root / "pitch_single_frequency_pareto_summary.csv"
    _write_csv(output_path, marked)
    figure_path = _plot_pareto(marked, output_root)
    return {
        "design_count": len(marked),
        "pareto_count": sum(1 for row in marked if row["is_pareto"]),
        "summary_path": output_path,
        "figure_path": figure_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mark Pareto candidates for the fixed-frequency pitch-stiffness scan.",
    )
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-shear", type=float, default=None)
    parser.add_argument("--max-bending", type=float, default=None)
    parser.add_argument("--max-released-moment", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pareto(args)
    print(f"design_count: {result['design_count']}")
    print(f"pareto_count: {result['pareto_count']}")
    print(f"summary_path: {result['summary_path']}")
    print(f"figure_path: {result['figure_path']}")


if __name__ == "__main__":
    main()
