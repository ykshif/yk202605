"""Combine multiple boundary18 DOE summaries and mark global Pareto rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import MetricObjective, mark_pareto_rows  # noqa: E402


DEFAULT_INPUTS = (
    REPO_ROOT / "results" / "boundary18_doe_single_frequency" / "boundary18_doe_summary.csv",
    REPO_ROOT / "results" / "boundary18_refined_single_frequency" / "boundary18_doe_summary.csv",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "boundary18_combined_single_frequency"

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
    "solve_elapsed_s",
    "sample_index",
    "design_dimension",
    "boundary_stiffness_min",
    "boundary_stiffness_max",
    "boundary_stiffness_mean",
}


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            row = dict(row)
            row["source_summary"] = str(path)
            row["sample_set"] = path.parent.name
            for field in NUMERIC_FIELDS:
                if field in row and row[field] != "":
                    row[field] = float(row[field])
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], output_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows])
    pareto = np.array([row["is_pareto"] in (True, "True", "true", "1") for row in rows])
    labels = [str(row["design_label"]) for row in rows]
    sample_sets = [str(row["sample_set"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.7), constrained_layout=True)
    for sample_set, marker in (
        ("boundary18_doe_single_frequency", "o"),
        ("boundary18_refined_single_frequency", "s"),
    ):
        indices = np.array([item == sample_set for item in sample_sets])
        if not np.any(indices):
            continue
        ax.scatter(
            mean_heave[indices & ~pareto],
            bending[indices & ~pareto],
            s=42,
            marker=marker,
            color="#adb5bd",
            label=f"{sample_set} dominated",
        )
        ax.scatter(
            mean_heave[indices & pareto],
            bending[indices & pareto],
            s=70,
            marker=marker,
            label=f"{sample_set} Pareto",
        )

    if np.any(pareto):
        order = np.argsort(mean_heave[pareto])
        ax.plot(mean_heave[pareto][order], bending[pareto][order], color="#212529", linewidth=1.1)

    for label, x_value, y_value, is_pareto in zip(labels, mean_heave, bending, pareto):
        if is_pareto:
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(5, 4))

    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max connector bending envelope")
    ax.set_title("combined boundary18 DOE Pareto screening")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "boundary18_combined_mean_heave_bending_pareto.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def run(args: argparse.Namespace) -> dict[str, object]:
    input_paths = tuple(Path(path).resolve() for path in args.inputs)
    rows: list[dict] = []
    for path in input_paths:
        rows.extend(_read_rows(path))

    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective("connector_bending", "max_connector_bending_envelope", minimize=True),
    )
    marked = mark_pareto_rows(rows, objectives)
    output_root = Path(args.output_root).resolve()
    summary_path = output_root / "boundary18_combined_pareto_summary.csv"
    _write_csv(summary_path, marked)
    figure_path = _plot(marked, output_root)
    return {
        "input_count": len(input_paths),
        "sample_count": len(marked),
        "pareto_count": sum(1 for row in marked if row["is_pareto"]),
        "summary_path": summary_path,
        "figure_path": figure_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine boundary18 DOE summaries and mark global Pareto rows.",
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
