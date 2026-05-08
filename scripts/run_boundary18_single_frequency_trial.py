"""Run one 18-variable boundary-stiffness trial for the 10x10 hinge case."""

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
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    evaluate_design_response,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
    solve_complex_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "boundary18_single_frequency_trial"


def center_stiff_profile(
    *,
    low: float = 1.0e8,
    high: float = 1.0e9,
) -> tuple[float, ...]:
    """Return 18 values, with middle internal boundaries stiffer."""

    values: list[float] = []
    center = 5
    max_distance = 4
    for _orientation in ("x", "y"):
        for boundary_index in range(1, 10):
            distance = abs(boundary_index - center)
            weight = 1.0 - distance / max_distance
            values.append(low * (high / low) ** weight)
    return tuple(values)


def uniform_profile(value: float) -> tuple[float, ...]:
    """Return 18 identical boundary stiffness values."""

    return tuple(float(value) for _ in range(18))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _design_values(args: argparse.Namespace) -> tuple[float, ...]:
    if args.profile == "uniform":
        return uniform_profile(args.uniform_value)
    if args.profile == "center_stiff":
        return center_stiff_profile(low=args.low, high=args.high)
    raise ValueError(f"Unsupported profile: {args.profile}")


def run_trial(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    case = build_complex_hinge_10x10_case(
        args.data_root,
        k_hinge=args.coupling_stiffness,
    )
    groups = build_hinge_design_groups(case, "continuous_boundary")
    design = BoundaryStiffnessDesign(
        values=_design_values(args),
        grouping="continuous_boundary",
        parameter="released_dof_stiffness",
        coupling_stiffness=args.coupling_stiffness,
        label=args.profile,
    )
    value_by_group = design.values_for_groups([group.name for group in groups])
    design_case = apply_grouped_hinge_stiffness(
        case,
        groups,
        value_by_group,
        parameter="released_dof_stiffness",
    )

    design_rows = [
        {
            "group_name": group.name,
            "orientation": group.orientation,
            "released_dof_stiffness": value_by_group[group.name],
            "hinge_line_count": len(group.hinge_indices),
            "hinge_lines": " ".join(str(index + 1) for index in group.hinge_indices),
        }
        for group in groups
    ]
    design_path = output_root / "boundary18_design_values.csv"
    _write_csv(design_path, design_rows)

    if args.response is not None:
        response = np.load(args.response)
        heave_grid = np.load(args.heave_grid) if args.heave_grid is not None else None
        omega = args.omega
        elapsed = 0.0
    elif args.solve:
        start = time.perf_counter()
        solved = solve_complex_hinge_case(design_case)
        elapsed = time.perf_counter() - start
        response = solved.response
        heave_grid = solved.heave_grid_merged
        omega = solved.omega
        np.save(output_root / "response_boundary18.npy", response)
        np.save(output_root / "heave_grid_boundary18.npy", heave_grid)
    else:
        return {
            "status": "design_written",
            "design_path": design_path,
            "note": "Run with --solve or --response to compute metrics.",
        }

    evaluation = evaluate_design_response(
        design_case,
        response,
        omega,
        design=design.as_dict([group.name for group in groups]),
        scenario={
            "omega": omega,
            "frequency_index": args.frequency_index,
            "wave_direction_deg": 0.0,
            "scenario_label": "boundary18_single_frequency",
        },
        heave_grid=heave_grid,
        cid_prefix="boundary18",
    )

    summary_path = output_root / "boundary18_trial_summary.csv"
    connector_path = output_root / "boundary18_connector_envelopes.csv"
    summary_row = evaluation.summary_row()
    summary_row["solve_elapsed_s"] = elapsed
    _write_csv(summary_path, [summary_row])
    _write_csv(connector_path, list(evaluation.connector_rows))

    return {
        "status": "evaluated",
        "summary_path": summary_path,
        "connector_path": connector_path,
        "design_path": design_path,
        "solve_elapsed_s": elapsed,
        **evaluation.metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one 18-variable boundary-stiffness trial at a single frequency.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--profile", choices=("center_stiff", "uniform"), default="center_stiff")
    parser.add_argument("--low", type=float, default=1.0e8)
    parser.add_argument("--high", type=float, default=1.0e9)
    parser.add_argument("--uniform-value", type=float, default=1.0e8)
    parser.add_argument("--coupling-stiffness", type=float, default=1.0e10)
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--omega", type=float, default=0.5851)
    parser.add_argument("--response", type=Path, default=None)
    parser.add_argument("--heave-grid", type=Path, default=None)
    parser.add_argument("--solve", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_trial(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
