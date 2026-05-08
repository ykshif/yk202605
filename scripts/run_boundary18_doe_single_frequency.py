"""Run a small single-frequency DOE for 18 boundary hinge stiffness variables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    BoundaryStiffnessDesign,
    MetricObjective,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    evaluate_design_response,
    generate_boundary18_doe_samples,
    generate_boundary18_refined_samples,
    mark_pareto_rows,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
    solve_complex_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "boundary18_doe_single_frequency"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_pareto(rows: list[dict], output_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows])
    pareto = np.array([row["is_pareto"] in (True, "True", "true", "1") for row in rows])
    labels = [str(row["design_label"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    if np.any(~pareto):
        ax.scatter(
            mean_heave[~pareto],
            bending[~pareto],
            s=54,
            color="#868e96",
            label="dominated",
        )
    if np.any(pareto):
        ax.scatter(
            mean_heave[pareto],
            bending[pareto],
            s=80,
            color="#1c7ed6",
            label="Pareto candidate",
        )
        order = np.argsort(mean_heave[pareto])
        ax.plot(mean_heave[pareto][order], bending[pareto][order], color="#1c7ed6", linewidth=1.2)

    for label, x_value, y_value, is_pareto in zip(labels, mean_heave, bending, pareto):
        if is_pareto or label.startswith(("uniform", "center", "edge")):
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 5))

    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max connector bending envelope")
    ax.set_title("boundary18 DOE Pareto screening at one frequency")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "boundary18_doe_mean_heave_bending_pareto.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def _top_connector_rows(connector_rows: tuple[dict, ...], *, design_label: str) -> list[dict]:
    """Return the controlling connector rows for one design."""

    keys = (
        ("max_shear", "shear_force_envelope"),
        ("max_bending", "bending_moment_envelope"),
        ("max_released", "released_moment_envelope"),
    )
    output = []
    for rank_name, key in keys:
        row = max(connector_rows, key=lambda item: float(item[key]))
        output.append(
            {
                "design_label": design_label,
                "rank_name": rank_name,
                **row,
            }
        )
    return output


def run_doe(args: argparse.Namespace) -> dict[str, object]:
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
    if args.sample_set == "initial":
        samples = generate_boundary18_doe_samples(
            low=args.low,
            high=args.high,
            random_count=args.random_count,
            seed=args.seed,
        )
    elif args.sample_set == "refined":
        samples = generate_boundary18_refined_samples(
            low=args.low,
            high=args.high,
        )
    else:
        raise ValueError(f"Unsupported sample set: {args.sample_set}")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    design_rows: list[dict] = []
    summary_rows: list[dict] = []
    top_connector_rows: list[dict] = []

    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] solving {sample.name}", flush=True)
        design = BoundaryStiffnessDesign(
            values=sample.value_by_group(),
            grouping="continuous_boundary",
            parameter="released_dof_stiffness",
            coupling_stiffness=args.coupling_stiffness,
            label=sample.name,
            meta={"sample_description": sample.description},
        )
        value_by_group = design.values_for_groups(group_names)
        case = apply_grouped_hinge_stiffness(
            base_case,
            groups,
            value_by_group,
            parameter="released_dof_stiffness",
        )

        for group in groups:
            design_rows.append(
                {
                    "design_label": sample.name,
                    "sample_description": sample.description,
                    "group_name": group.name,
                    "orientation": group.orientation,
                    "released_dof_stiffness": value_by_group[group.name],
                    "hinge_line_count": len(group.hinge_indices),
                    "hinge_lines": " ".join(str(index + 1) for index in group.hinge_indices),
                }
            )

        start = time.perf_counter()
        solved = solve_complex_hinge_case(case)
        elapsed = time.perf_counter() - start

        np.save(response_root / f"response_{sample.name}.npy", solved.response)
        np.save(response_root / f"heave_grid_{sample.name}.npy", solved.heave_grid_merged)
        evaluation = evaluate_design_response(
            case,
            solved.response,
            solved.omega,
            design=design.as_dict(group_names),
            scenario={
                "omega": solved.omega,
                "frequency_index": args.frequency_index,
                "wave_direction_deg": 0.0,
                "scenario_label": f"boundary18_{args.sample_set}_single_frequency",
            },
            heave_grid=solved.heave_grid_merged,
            cid_prefix=sample.name,
        )

        summary_row = evaluation.summary_row()
        summary_row["sample_index"] = sample_index
        summary_row["solve_elapsed_s"] = elapsed
        summary_row["response_path"] = str(response_root / f"response_{sample.name}.npy")
        summary_row["heave_grid_path"] = str(response_root / f"heave_grid_{sample.name}.npy")
        summary_rows.append(summary_row)
        top_connector_rows.extend(_top_connector_rows(evaluation.connector_rows, design_label=sample.name))

    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective("connector_bending", "max_connector_bending_envelope", minimize=True),
    )
    pareto_rows = mark_pareto_rows(summary_rows, objectives)

    design_values_path = output_root / "boundary18_doe_design_values.csv"
    summary_path = output_root / "boundary18_doe_summary.csv"
    pareto_path = output_root / "boundary18_doe_pareto_summary.csv"
    top_connectors_path = output_root / "boundary18_doe_top_connectors.csv"
    _write_csv(design_values_path, design_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(pareto_path, pareto_rows)
    _write_csv(top_connectors_path, top_connector_rows)
    figure_path = _plot_pareto(pareto_rows, output_root)

    return {
        "sample_count": len(samples),
        "pareto_count": sum(1 for row in pareto_rows if row["is_pareto"]),
        "summary_path": summary_path,
        "pareto_path": pareto_path,
        "figure_path": figure_path,
        "design_values_path": design_values_path,
        "top_connectors_path": top_connectors_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small DOE for 18 boundary stiffness variables at one frequency.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--low", type=float, default=1.0e8)
    parser.add_argument("--high", type=float, default=1.0e9)
    parser.add_argument("--sample-set", choices=("initial", "refined"), default="initial")
    parser.add_argument("--random-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--coupling-stiffness", type=float, default=1.0e10)
    parser.add_argument("--frequency-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_doe(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
