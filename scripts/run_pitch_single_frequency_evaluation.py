"""Run fixed-frequency pitch-stiffness design evaluation from cached responses.

The script intentionally performs a parameter scan, not an optimization. It
reuses solved 10x10 responses and applies the common evaluator from
``offshore_energy_sim.optimization`` to produce response/connector metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import evaluate_design  # noqa: E402


DEFAULT_RECORDS_PATH = (
    REPO_ROOT
    / "results"
    / "complex_hinge_10x10_pitch_stiffness_sweep"
    / "centerline_records.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "single_frequency_pitch_design_evaluation"


def _load_records(records_path: Path) -> list[dict]:
    with records_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    records = payload.get("records", payload)
    if not isinstance(records, list):
        raise ValueError("records file must contain a list or {'records': list}")
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary_rows: list[dict], output_root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(row["pitch_stiffness_label"]) for row in summary_rows]
    x = np.arange(len(labels))
    max_heave = np.array([float(row["max_heave"]) for row in summary_rows])
    mean_heave = np.array([float(row["mean_heave"]) for row in summary_rows])
    max_shear = np.array([float(row["max_connector_shear_envelope"]) for row in summary_rows])
    max_bending = np.array(
        [float(row["max_connector_bending_envelope"]) for row in summary_rows]
    )
    max_released = np.array([float(row["max_released_moment_envelope"]) for row in summary_rows])

    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(x, max_heave, marker="o", label="max heave")
    axes[0, 0].plot(x, mean_heave, marker="s", label="mean heave")
    axes[0, 0].set_ylabel("heave amplitude (m)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, max_shear, marker="o", color="#0b7285")
    axes[0, 1].set_ylabel("max shear envelope")

    axes[1, 0].plot(x, max_bending, marker="o", color="#c92a2a")
    axes[1, 0].set_ylabel("max bending moment envelope")

    axes[1, 1].plot(x, max_released, marker="o", color="#5f3dc4")
    axes[1, 1].set_ylabel("max released moment envelope")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("pitch stiffness")
        ax.grid(True, color="#d9d9d9", linewidth=0.7)

    tradeoff_path = output_root / "pitch_single_frequency_evaluation_summary.png"
    fig.savefig(tradeoff_path, dpi=220)
    plt.close(fig)
    paths.append(tradeoff_path)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.scatter(mean_heave, max_bending, s=64, color="#c92a2a")
    for label, x_value, y_value in zip(labels, mean_heave, max_bending):
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 5))
    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max bending moment envelope")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    pareto_path = output_root / "pitch_single_frequency_heave_bending_tradeoff.png"
    fig.savefig(pareto_path, dpi=220)
    plt.close(fig)
    paths.append(pareto_path)
    return paths


def run_scan(args: argparse.Namespace) -> dict[str, Path | int]:
    records_path = Path(args.records).resolve()
    output_root = Path(args.output_root).resolve()
    records = _load_records(records_path)
    source_root = records_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for record in records:
        label = str(record["pitch_stiffness_label"])
        response_path = source_root / f"response_k_pitch{label}.npy"
        heave_grid_path = source_root / f"heave_grid_merged_k_pitch{label}.npy"
        if not response_path.exists() and not args.solve_missing:
            raise FileNotFoundError(f"Missing cached response: {response_path}")

        response = np.load(response_path) if response_path.exists() else None
        heave_grid = np.load(heave_grid_path) if heave_grid_path.exists() else None
        evaluation = evaluate_design(
            {
                "pitch_stiffness": record["pitch_stiffness"],
                "pitch_stiffness_label": label,
                "coupling_stiffness": record.get("fixed_coupling_stiffness", 1.0e10),
            },
            {
                "omega": record["omega"],
                "frequency_index": record.get("frequency_index", 0),
                "wave_direction_deg": args.wave_direction_deg,
                "scenario_label": "cached_single_frequency",
            },
            data_root=args.data_root,
            response=response,
            heave_grid=heave_grid,
            solve_if_response_missing=args.solve_missing,
            cid_prefix="pitch_scan",
        )

        detail_path = output_root / f"connector_envelopes_k_pitch_{label}.csv"
        _write_csv(detail_path, list(evaluation.connector_rows))
        summary_row = evaluation.summary_row()
        summary_row["detail_csv"] = str(detail_path)
        summary_row["source_response_path"] = str(response_path)
        summary_row["source_heave_grid_path"] = str(heave_grid_path)
        summary_rows.append(summary_row)

    summary_path = output_root / "pitch_single_frequency_evaluation_summary.csv"
    _write_csv(summary_path, summary_rows)
    figure_paths = _plot_summary(summary_rows, output_root)
    return {
        "design_count": len(summary_rows),
        "summary_path": summary_path,
        "figure_count": len(figure_paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-frequency 10x10 pitch-stiffness designs from cached responses.",
    )
    parser.add_argument("--records", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--wave-direction-deg", type=float, default=0.0)
    parser.add_argument(
        "--solve-missing",
        action="store_true",
        help="Solve the 10x10 case if a cached response is missing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_scan(args)
    print(f"design_count: {result['design_count']}")
    print(f"summary_path: {result['summary_path']}")
    print(f"figure_count: {result['figure_count']}")


if __name__ == "__main__":
    main()
