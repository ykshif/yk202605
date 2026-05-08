"""Compare 5x5, 10x10, and 15x15 modular hinge discretizations.

The script has three roles for the paper workflow:

1. inventory the design-space and input files for each module count;
2. optionally generate missing Capytaine hydrodynamic NetCDF files through the
   same standardized backend used by the local hydrodynamics UI;
3. solve available RODM hinge cases and write publication-oriented PDF figures.

By default the script solves only cases with complete input files.  For
publication-quality 5x5 and 15x15 response comparisons, matching structural
mass/stiffness matrices for 60 m and 20 m modules should be supplied explicitly
with ``--structure-map``.  The existing 30 m structure matrix is used directly
only for the validated 10x10 case unless ``--allow-shared-30m-structure`` is
given.
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

from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    run_array_hydrodynamics,
)
from offshore_energy_sim.optimization import (  # noqa: E402
    evaluate_design_response,
    summarize_hinge_design_space,
)
from offshore_energy_sim.structure import scan_abaqus_matrix_file  # noqa: E402
from offshore_energy_sim.validation import (  # noqa: E402
    build_modular_hinge_grid_case,
    default_hydrodynamic_output_path,
    default_module_size_m,
    solve_complex_hinge_case,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import missing_input_paths  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "module_count_comparison"
DEFAULT_DATA_ROOT = Path("/Users/yongkang/data/DM-FEM2D")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{float(value):.2e}".replace("+", "").replace(".", "p")


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = []
    for token in text.replace(";", ",").split(","):
        stripped = token.strip()
        if stripped:
            values.append(float(stripped))
    return tuple(values)


def _infer_nodes_per_module_side(matrix_path: Path) -> int:
    summary = scan_abaqus_matrix_file(matrix_path)
    if summary.max_node_id is None:
        raise ValueError(f"Cannot infer node count from {matrix_path}")
    side = int(round(summary.max_node_id**0.5))
    if side * side != summary.max_node_id:
        raise ValueError(
            f"Structure matrix {matrix_path} has {summary.max_node_id} nodes, not a square module mesh"
        )
    return side


def _load_structure_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        n = int(key)
        mass_path = Path(value["mass_matrix_path"])
        stiffness_path = Path(value["stiffness_matrix_path"])
        nodes_per_side = value.get("nodes_per_module_side")
        if nodes_per_side is None:
            nodes_per_side = _infer_nodes_per_module_side(mass_path)
        center_node = value.get("center_node_one_based")
        if center_node is None:
            center_node = (int(nodes_per_side) ** 2 + 1) // 2
        output[n] = {
            "mass_matrix_path": mass_path,
            "stiffness_matrix_path": stiffness_path,
            "nodes_per_module_side": int(nodes_per_side),
            "center_node_one_based": int(center_node),
        }
    return output


def _case_for_n(
    n: int,
    *,
    data_root: Path,
    total_size_m: float,
    released_dof_stiffness: float,
    structure_map: dict[int, dict[str, Any]],
    allow_shared_30m_structure: bool,
    hydrodynamic_path: Path | None = None,
):
    mass_path = stiffness_path = None
    nodes_per_module_side = 7
    center_node_one_based = None
    if n in structure_map:
        item = structure_map[n]
        mass_path = item["mass_matrix_path"]
        stiffness_path = item["stiffness_matrix_path"]
        nodes_per_module_side = int(item["nodes_per_module_side"])
        center_node_one_based = int(item["center_node_one_based"])
    elif n != 10 and not allow_shared_30m_structure:
        # Use deliberately impossible placeholders so inventory records the
        # missing publication-quality structural inputs instead of silently
        # reusing the 30 m matrix for a 60 m or 20 m module.
        module_size = default_module_size_m(n, total_size_m=total_size_m)
        root = data_root / "StructureData" / "ModularGrid"
        label = f"{module_size:.6g}".replace(".", "p")
        mass_path = root / f"Job{label}m_module_MASS1.mtx"
        stiffness_path = root / f"Job{label}m_module_STIF1.mtx"
    return build_modular_hinge_grid_case(
        n,
        data_root,
        total_size_m=total_size_m,
        nodes_per_module_side=nodes_per_module_side,
        center_node_one_based=center_node_one_based,
        released_dof_stiffness=released_dof_stiffness,
        mass_matrix_path=mass_path,
        stiffness_matrix_path=stiffness_path,
        hydrodynamic_path=hydrodynamic_path,
    )


def _hydrodynamic_config_for_n(
    n: int,
    *,
    data_root: Path,
    total_size_m: float,
    output_path: Path,
    omega: float,
    direction_deg: float,
    panels_per_side: float,
    vertical_panels: float,
    rho: float,
    draft_m: float,
    height_m: float,
    n_jobs: int,
    compute_rao: bool,
) -> ArrayHydrodynamicsConfig:
    module_size = default_module_size_m(n, total_size_m=total_size_m)
    mesh_size = module_size / panels_per_side
    vertical_mesh_size = height_m / vertical_panels
    return ArrayHydrodynamicsConfig(
        module=RectangularModuleSpec(
            length_m=module_size,
            width_m=module_size,
            height_m=height_m,
            draft_m=draft_m,
            mesh_size_m=mesh_size,
            vertical_mesh_size_m=vertical_mesh_size,
            mass_kg=module_size * module_size * draft_m * rho,
        ),
        layout=ArrayLayoutSpec(
            rows=n,
            columns=n,
            spacing_x_m=module_size + 0.01,
            spacing_y_m=module_size + 0.01,
        ),
        omegas_rad_s=(omega,),
        wave_directions_rad=(float(np.deg2rad(direction_deg)),),
        output_path=output_path,
        water_depth_m=None,
        rho=rho,
        g=9.81,
        n_jobs=n_jobs,
        compute_rao=compute_rao,
    )


def _plot_design_space(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n = np.array([int(row["modules_per_side"]) for row in rows])
    module_count = np.array([int(row["module_count"]) for row in rows])
    hinge_lines = np.array([int(row["hinge_line_count"]) for row in rows])
    boundary_dim = np.array([int(row["continuous_boundary_dimension"]) for row in rows])
    segment_dim = np.array([int(row["segment_line_dimension"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(n, module_count, marker="o", label="modules")
    ax.plot(n, hinge_lines, marker="s", label="hinge lines")
    ax.plot(n, segment_dim, marker="^", label="line-level variables")
    ax.plot(n, boundary_dim, marker="D", label="boundary-level variables")
    ax.set_xlabel("modules per side")
    ax.set_ylabel("count")
    ax.set_title("Module-count design-space growth")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    path = output_root / "module_count_design_space_growth.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    paths.append(path)

    connector_pairs = np.array([int(row["connector_pair_count"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.bar([str(value) for value in n], connector_pairs, color="#1971c2")
    ax.set_xlabel("modules per side")
    ax.set_ylabel("connector node pairs")
    ax.set_title("Connector-pair count for module discretizations")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    path = output_root / "module_count_connector_pairs.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_response_summary(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = [f"{int(row['modules_per_side'])}x{int(row['modules_per_side'])}\n{row['stiffness_label']}" for row in rows]
    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows])
    shear = np.array([float(row["max_connector_shear_envelope"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), constrained_layout=True)
    axes[0].bar(labels, mean_heave, color="#2f9e44")
    axes[0].set_ylabel("mean heave amplitude (m)")
    axes[1].bar(labels, bending, color="#e67700")
    axes[1].set_ylabel("max connector bending envelope")
    axes[2].bar(labels, shear, color="#5f3dc4")
    axes[2].set_ylabel("max connector shear envelope")
    for ax in axes:
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Solved module-count response metrics")
    path = output_root / "module_count_solved_response_metrics.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    paths.append(path)
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    figure_root = output_root / "figures"
    response_root = output_root / "responses"
    output_root.mkdir(parents=True, exist_ok=True)
    response_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    module_counts = tuple(int(value) for value in args.module_counts.split(","))
    stiffness_values = _parse_float_list(args.stiffness_values)
    structure_map = _load_structure_map(Path(args.structure_map) if args.structure_map else None)

    inventory_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    generated_hydro_rows: list[dict[str, Any]] = []

    for n in module_counts:
        generated_hydro_path = default_hydrodynamic_output_path(
            n,
            data_root=data_root,
            total_size_m=args.total_size_m,
            omega=args.omega,
            direction_deg=args.direction_deg,
        )
        if args.generate_missing_hydro and not generated_hydro_path.exists():
            print(f"[hydro] generating {n}x{n}: {generated_hydro_path}", flush=True)
            config = _hydrodynamic_config_for_n(
                n,
                data_root=data_root,
                total_size_m=args.total_size_m,
                output_path=generated_hydro_path,
                omega=args.omega,
                direction_deg=args.direction_deg,
                panels_per_side=args.panels_per_side,
                vertical_panels=args.vertical_panels,
                rho=args.rho,
                draft_m=args.draft_m,
                height_m=args.height_m,
                n_jobs=args.n_jobs,
                compute_rao=not args.skip_rao,
            )
            start = time.perf_counter()
            result = run_array_hydrodynamics(config, log=lambda message: print(f"[hydro:{n}x{n}] {message}", flush=True))
            generated_hydro_rows.append(
                {
                    "modules_per_side": n,
                    "hydrodynamic_path": str(result.output_path),
                    "body_count": result.body_count,
                    "problem_count": result.problem_count,
                    "elapsed_s": time.perf_counter() - start,
                }
            )

        for stiffness in stiffness_values:
            case = _case_for_n(
                n,
                data_root=data_root,
                total_size_m=args.total_size_m,
                released_dof_stiffness=stiffness,
                structure_map=structure_map,
                allow_shared_30m_structure=args.allow_shared_30m_structure,
                hydrodynamic_path=generated_hydro_path if generated_hydro_path.exists() else None,
            )
            summary = summarize_hinge_design_space(case)
            missing = missing_input_paths(case)
            inventory_row = {
                "modules_per_side": n,
                "module_size_m": case.grid.module_size,
                "total_size_m": case.grid.structure_size,
                "nodes_per_module_side": case.grid.nodes_per_module_side,
                "nodes_per_module": case.grid.nodes_per_module,
                "center_node_one_based": case.grid.center_node_one_based,
                "module_count": case.grid.module_count,
                "structural_node_count": case.grid.total_nodes,
                "retained_structural_dof_count": case.grid.total_nodes
                * case.retained_dofs_per_node,
                "hydrodynamic_node_count": case.hydrodynamic_nodes,
                "retained_hydrodynamic_dof_count": case.hydrodynamic_nodes
                * case.retained_dofs_per_node,
                "single_frequency_bem_problem_count": case.hydrodynamic_nodes * 6 + 1,
                "hinge_line_count": summary.hinge_line_count,
                "x_hinge_line_count": summary.x_hinge_line_count,
                "y_hinge_line_count": summary.y_hinge_line_count,
                "connector_pair_count": summary.connector_pair_count,
                "pairs_per_hinge_line": summary.pairs_per_hinge_line,
                "continuous_boundary_dimension": summary.continuous_boundary_dimension,
                "segment_line_dimension": summary.segment_line_dimension,
                "connector_pair_dimension": summary.connector_pair_dimension,
                "released_dof_stiffness": stiffness,
                "stiffness_label": _safe_label(stiffness),
                "mass_matrix_path": str(case.mass_matrix_path),
                "stiffness_matrix_path": str(case.stiffness_matrix_path),
                "hydrodynamic_path": str(case.hydrodynamic_path),
                "mass_matrix_exists": case.mass_matrix_path.exists(),
                "stiffness_matrix_exists": case.stiffness_matrix_path.exists(),
                "hydrodynamic_exists": case.hydrodynamic_path.exists(),
                "input_complete": not missing,
                "missing_inputs": "; ".join(str(path) for path in missing),
            }
            inventory_rows.append(inventory_row)
            if args.solve and not missing:
                print(f"[solve] {n}x{n}, k={stiffness:g}", flush=True)
                start = time.perf_counter()
                solved = solve_complex_hinge_case(case)
                elapsed = time.perf_counter() - start
                label = f"{n}x{n}_k{_safe_label(stiffness)}"
                response_path = response_root / f"response_{label}.npy"
                heave_path = response_root / f"heave_grid_{label}.npy"
                np.save(response_path, solved.response)
                np.save(heave_path, solved.heave_grid_merged)
                evaluation = evaluate_design_response(
                    case,
                    solved.response,
                    solved.omega,
                    design={
                        "modules_per_side": n,
                        "module_size_m": case.grid.module_size,
                        "released_dof_stiffness": stiffness,
                        "stiffness_label": _safe_label(stiffness),
                    },
                    scenario={
                        "omega": solved.omega,
                        "frequency_index": case.frequency_index,
                        "wave_direction_deg": args.direction_deg,
                        "scenario_label": "module_count_comparison",
                    },
                    heave_grid=solved.heave_grid_merged,
                    cid_prefix=label,
                )
                row = evaluation.summary_row()
                row.update(
                    {
                        "solve_elapsed_s": elapsed,
                        "response_path": str(response_path),
                        "heave_grid_path": str(heave_path),
                    }
                )
                response_rows.append(row)

    inventory_path = output_root / "module_count_design_space_and_inputs.csv"
    _write_csv(inventory_path, inventory_rows)
    response_path = None
    if response_rows:
        response_path = output_root / "module_count_response_summary.csv"
        _write_csv(response_path, response_rows)
    generated_hydro_path = None
    if generated_hydro_rows:
        generated_hydro_path = output_root / "module_count_generated_hydrodynamics.csv"
        _write_csv(generated_hydro_path, generated_hydro_rows)

    design_figures = _plot_design_space(
        [
            row
            for row in inventory_rows
            if float(row["released_dof_stiffness"]) == stiffness_values[0]
        ],
        figure_root,
    )
    response_figures = _plot_response_summary(response_rows, figure_root)

    result = {
        "inventory_path": str(inventory_path),
        "response_path": None if response_path is None else str(response_path),
        "generated_hydro_path": None if generated_hydro_path is None else str(generated_hydro_path),
        "design_figures": [str(path) for path in design_figures],
        "response_figures": [str(path) for path in response_figures],
        "inventory_rows": len(inventory_rows),
        "response_rows": len(response_rows),
        "generated_hydro_rows": len(generated_hydro_rows),
    }
    (output_root / "module_count_comparison_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--module-counts", default="5,10,15")
    parser.add_argument("--total-size-m", type=float, default=300.0)
    parser.add_argument("--omega", type=float, default=0.5851)
    parser.add_argument("--direction-deg", type=float, default=0.0)
    parser.add_argument("--stiffness-values", default="0,1e7,1e9")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--generate-missing-hydro", action="store_true")
    parser.add_argument("--allow-shared-30m-structure", action="store_true")
    parser.add_argument("--structure-map", default="")
    parser.add_argument("--panels-per-side", type=float, default=5.0)
    parser.add_argument("--vertical-panels", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--draft-m", type=float, default=1.1)
    parser.add_argument("--height-m", type=float, default=4.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-rao", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
