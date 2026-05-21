"""Run a SEREP-ridge non-uniform module design study.

The study keeps the physical floating body fixed and varies only the 1D module
length distribution along x. All RODM solves use the repaired SEREP variant:
``structural_reduction_method="serep_ridge"`` with preserved master-node order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import MasterNodeRule, RodmFrequencyCase, StructuralMatrixPaths  # noqa: E402
from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    module_structural_node_mappings,
    omega_values_from_wavelengths,
    run_array_hydrodynamics,
)
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


LENGTH_M = 300.0
WIDTH_M = 60.0
HEIGHT_M = 2.0
DRAFT_M = 0.5
WATER_DEPTH_M = 58.5
RHO = 1000.0
G = 9.81
MESH_SIZE_M = 2.0
WAVELENGTHS_M = (60, 120, 180, 240, 300)
CAPYTAINE_N_JOBS = min(16, max(1, (os.cpu_count() or 1) - 2))
SEREP_RIDGE_RELATIVE_LAMBDA = 1.0e-16

OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_design_study"
REFERENCE_ROOT = REPO_ROOT / "results" / "uniform_reference_convergence_U5_U10_U15_U30_heave_serep_ridge_ordered"
STRUCTURE_DIR = Path(r"E:\phd\Code\DM-FEM2D") / "StructureData"


LAYOUTS: dict[str, tuple[float, ...]] = {
    "edge_mild": (20, 30, 30, 40, 30, 30, 40, 30, 30, 20),
    "edge_strong": (20, 20, 30, 40, 40, 40, 40, 30, 20, 20),
    "bow_refined": (20, 20, 20, 30, 30, 30, 30, 40, 40, 40),
    "stern_refined": (40, 40, 40, 30, 30, 30, 30, 20, 20, 20),
    "center_refined": (40, 40, 30, 20, 20, 20, 20, 30, 40, 40),
    "alternating": (20, 40, 20, 40, 30, 30, 40, 20, 40, 20),
}


@dataclass(frozen=True)
class CaseResult:
    layout_id: str
    wavelength_m: int
    omega_rad_s: float
    rmse_vs_U30_serep_ridge: float
    max_abs_vs_U30_serep_ridge: float
    roughness: float
    response_path: Path
    hydro_path: Path
    figure_path: Path


def structural_paths() -> StructuralMatrixPaths:
    return StructuralMatrixPaths(
        mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
    )


def build_hydro_config(
    layout_id: str,
    module_lengths_m: tuple[float, ...],
    *,
    n_jobs: int,
) -> ArrayHydrodynamicsConfig:
    return ArrayHydrodynamicsConfig(
        module=RectangularModuleSpec(
            length_m=30.0,
            width_m=WIDTH_M,
            height_m=HEIGHT_M,
            draft_m=DRAFT_M,
            mesh_size_m=MESH_SIZE_M,
            vertical_mesh_size_m=MESH_SIZE_M,
        ),
        layout=ArrayLayoutSpec(
            rows=1,
            columns=len(module_lengths_m),
            spacing_x_m=30.0,
            spacing_y_m=WIDTH_M,
            division_mode="custom",
            total_length_m=LENGTH_M,
            module_lengths_x_m=module_lengths_m,
        ),
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, G),
        output_path=OUTPUT_ROOT
        / "hydro"
        / f"{layout_id}_U10_D0p5_rho1000_wl60_300_mesh2.nc",
        wave_directions_rad=(0.0,),
        water_depth_m=WATER_DEPTH_M,
        rho=RHO,
        g=G,
        n_jobs=n_jobs,
        compute_rao=False,
        structural_grid=StructuralGridSpec(length_m=LENGTH_M, width_m=WIDTH_M, dx_m=5.0, dy_m=5.0),
    )


def ensure_hydrodynamics(config: ArrayHydrodynamicsConfig, *, force: bool) -> None:
    if config.output_path.exists() and not force:
        return
    logs: list[str] = []
    start = time.perf_counter()
    run_array_hydrodynamics(config, log=logs.append)
    elapsed = time.perf_counter() - start
    config.output_path.with_suffix(".generation_log.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_case(
    *,
    layout_id: str,
    hydro_path: Path,
    master_nodes: tuple[int, ...],
    wavelength_index: int,
) -> RodmFrequencyCase:
    return RodmFrequencyCase(
        case_id=f"{layout_id}_{WAVELENGTHS_M[wavelength_index]}m_serep_ridge",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=master_nodes[0], node_interval=1, count=len(master_nodes)),
        master_nodes_one_based=master_nodes,
        hydrodynamic_dataset=hydro_path,
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=len(master_nodes),
        hydrodynamic_dof_to_remove_zero_based=5,
        structural_reduction_method="serep_ridge",
        preserve_master_order=True,
        serep_ridge_relative_lambda=SEREP_RIDGE_RELATIVE_LAMBDA,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=False,
    )


def reference_response_path(wavelength_m: int) -> Path:
    return REFERENCE_ROOT / "responses" / "U30" / f"uniform_U30_wavelength_{wavelength_m}m_response.npy"


def uniform_u10_response_path(wavelength_m: int) -> Path:
    return REFERENCE_ROOT / "responses" / "U10" / f"uniform_U10_wavelength_{wavelength_m}m_response.npy"


def roughness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.max(np.abs(np.diff(values, n=2))))


def write_geometry_csv(layout_id: str, config: ArrayHydrodynamicsConfig) -> Path:
    geometries = config.layout.module_geometries(config.module.length_m)
    mappings = module_structural_node_mappings(config)
    path = OUTPUT_ROOT / "geometry" / f"{layout_id}_module_geometry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (geometry, mapping) in enumerate(zip(geometries, mappings), start=1):
        rows.append(
            {
                "layout_id": layout_id,
                "module_id": index,
                "module_length_m": geometry.length_m,
                "x_start_m": geometry.x_start_m,
                "x_end_m": geometry.x_end_m,
                "center_x_m": geometry.x_m,
                "width_m": WIDTH_M,
                "selected_node_id": mapping["fem_node_one_based"],
                "selected_node_x_m": mapping["x_m"],
                "selected_node_y_m": mapping["y_m"],
                "abs_error_m": abs(float(mapping["x_m"]) - geometry.x_m),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def solve_layout(
    layout_id: str,
    config: ArrayHydrodynamicsConfig,
) -> list[CaseResult]:
    mappings = module_structural_node_mappings(config)
    master_nodes = tuple(int(item["fem_node_one_based"]) for item in mappings)
    omegas = tuple(float(value) for value in config.omegas_rad_s)
    results: list[CaseResult] = []
    for wavelength_index, wavelength_m in enumerate(WAVELENGTHS_M):
        response_path = OUTPUT_ROOT / "responses" / layout_id / f"{layout_id}_{wavelength_m}m_response.npy"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        case = build_case(
            layout_id=layout_id,
            hydro_path=config.output_path,
            master_nodes=master_nodes,
            wavelength_index=wavelength_index,
        )
        response = solve_rodm_frequency_case(case).global_displacement
        np.save(response_path, response)
        _, heave = extract_centerline_heave(response)
        reference = np.load(reference_response_path(wavelength_m))
        _, heave_ref = extract_centerline_heave(reference)
        delta = heave - heave_ref
        results.append(
            CaseResult(
                layout_id=layout_id,
                wavelength_m=wavelength_m,
                omega_rad_s=omegas[wavelength_index],
                rmse_vs_U30_serep_ridge=float(np.sqrt(np.mean(delta**2))),
                max_abs_vs_U30_serep_ridge=float(np.max(np.abs(delta))),
                roughness=roughness(heave),
                response_path=response_path,
                hydro_path=config.output_path,
                figure_path=OUTPUT_ROOT / "figures" / "per_wavelength" / f"{layout_id}_{wavelength_m}m.png",
            )
        )
    return results


def add_uniform_baseline() -> list[CaseResult]:
    results: list[CaseResult] = []
    for wavelength_m in WAVELENGTHS_M:
        response = np.load(uniform_u10_response_path(wavelength_m))
        reference = np.load(reference_response_path(wavelength_m))
        _, heave = extract_centerline_heave(response)
        _, heave_ref = extract_centerline_heave(reference)
        delta = heave - heave_ref
        results.append(
            CaseResult(
                layout_id="uniform_U10",
                wavelength_m=wavelength_m,
                omega_rad_s=float("nan"),
                rmse_vs_U30_serep_ridge=float(np.sqrt(np.mean(delta**2))),
                max_abs_vs_U30_serep_ridge=float(np.max(np.abs(delta))),
                roughness=roughness(heave),
                response_path=uniform_u10_response_path(wavelength_m),
                hydro_path=REFERENCE_ROOT / "hydro",
                figure_path=OUTPUT_ROOT / "figures" / "per_wavelength" / f"uniform_U10_{wavelength_m}m.png",
            )
        )
    return results


def plot_response_panel(layout_ids: list[str], results: list[CaseResult]) -> Path:
    import matplotlib.pyplot as plt

    response_by_layout_wl = {(item.layout_id, item.wavelength_m): item.response_path for item in results}
    colors = {
        "U30 reference": "#111111",
        "uniform_U10": "#1f77b4",
        "edge_mild": "#d62728",
        "edge_strong": "#ff7f0e",
        "bow_refined": "#2ca02c",
        "stern_refined": "#9467bd",
        "center_refined": "#8c564b",
        "alternating": "#17becf",
    }
    fig, axes = plt.subplots(len(WAVELENGTHS_M), 1, figsize=(11.0, 16.0), sharex=True)
    fig.suptitle("SEREP-ridge Non-uniform Module Study: Heave vs U30 Reference", fontsize=16)
    for axis, wavelength_m in zip(axes, WAVELENGTHS_M):
        reference = np.load(reference_response_path(wavelength_m))
        x, heave_ref = extract_centerline_heave(reference)
        axis.plot(x, heave_ref, color=colors["U30 reference"], linewidth=2.2, label="U30 reference")
        for layout_id in layout_ids:
            response = np.load(response_by_layout_wl[(layout_id, wavelength_m)])
            _, heave = extract_centerline_heave(response)
            linestyle = "-" if layout_id == "uniform_U10" else "--"
            axis.plot(
                x,
                heave,
                color=colors[layout_id],
                linewidth=1.4,
                linestyle=linestyle,
                label=layout_id,
                alpha=0.95,
            )
        axis.set_ylabel(f"{wavelength_m} m\nHeave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("x/L")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = OUTPUT_ROOT / "figures" / "serep_ridge_nonuniform_heave_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_error_summary(layout_ids: list[str], results: list[CaseResult]) -> Path:
    import matplotlib.pyplot as plt

    rows_by_layout = {layout_id: [item for item in results if item.layout_id == layout_id] for layout_id in layout_ids}
    mean_rmse = {
        layout_id: float(np.mean([item.rmse_vs_U30_serep_ridge for item in rows]))
        for layout_id, rows in rows_by_layout.items()
    }
    sorted_layouts = sorted(layout_ids, key=lambda item: mean_rmse[item])
    x = np.arange(len(sorted_layouts))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    axes[0].bar(x, [mean_rmse[item] for item in sorted_layouts], color="#1f77b4")
    axes[0].set_title("Mean RMSE Across Wavelengths")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_layouts, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE")

    wavelengths = np.asarray(WAVELENGTHS_M, dtype=float)
    for layout_id in sorted_layouts:
        rows = rows_by_layout[layout_id]
        axes[1].plot(
            wavelengths,
            [item.rmse_vs_U30_serep_ridge for item in rows],
            marker="o",
            linewidth=1.4,
            label=layout_id,
        )
    axes[1].set_title("RMSE by Wavelength")
    axes[1].set_xlabel("wavelength (m)")
    axes[1].set_xticks(wavelengths)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    fig.suptitle("SEREP-ridge Non-uniform Module Error Ranking", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT_ROOT / "figures" / "serep_ridge_nonuniform_error_ranking.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_summary(
    *,
    results: list[CaseResult],
    geometry_paths: dict[str, Path],
    response_panel: Path,
    error_summary: Path,
) -> None:
    summary_csv = OUTPUT_ROOT / "serep_ridge_nonuniform_design_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CaseResult.__dataclass_fields__))
        writer.writeheader()
        for item in results:
            row = item.__dict__.copy()
            for key in ("response_path", "hydro_path", "figure_path"):
                row[key] = str(row[key])
            writer.writerow(row)

    layout_ids = list(dict.fromkeys(item.layout_id for item in results))
    ranked = []
    for layout_id in layout_ids:
        rows = [item for item in results if item.layout_id == layout_id]
        ranked.append(
            {
                "layout_id": layout_id,
                "mean_rmse": float(np.mean([item.rmse_vs_U30_serep_ridge for item in rows])),
                "mean_max_abs": float(np.mean([item.max_abs_vs_U30_serep_ridge for item in rows])),
            }
        )
    ranked.sort(key=lambda item: item["mean_rmse"])
    ranked_csv = OUTPUT_ROOT / "serep_ridge_nonuniform_layout_ranking.csv"
    with ranked_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)

    manifest = {
        "parameters": {
            "method": "serep_ridge",
            "preserve_master_order": True,
            "serep_ridge_relative_lambda": SEREP_RIDGE_RELATIVE_LAMBDA,
            "module_length_candidates_m": [20, 30, 40],
            "wavelengths_m": list(WAVELENGTHS_M),
        },
        "layouts": {
            "uniform_U10": [30.0] * 10,
            "uniform_U30_reference": [10.0] * 30,
            **{key: list(value) for key, value in LAYOUTS.items()},
        },
        "geometry_csv": {key: str(value) for key, value in geometry_paths.items()},
        "summary_csv": str(summary_csv),
        "ranking_csv": str(ranked_csv),
        "response_panel": str(response_panel),
        "error_summary": str(error_summary),
    }
    (OUTPUT_ROOT / "serep_ridge_nonuniform_design_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SEREP-ridge Non-uniform Module Design Study",
        "",
        "All cases use ordered SEREP-ridge reduction and are compared against the U30 SEREP-ridge reference.",
        "",
        f"- summary CSV: `{summary_csv}`",
        f"- ranking CSV: `{ranked_csv}`",
        f"- response panel: `{response_panel}`",
        f"- error summary: `{error_summary}`",
        "",
        "## Layout Ranking",
        "",
        "| rank | layout | mean RMSE | mean max abs | lengths m |",
        "| ---: | :--- | ---: | ---: | :--- |",
    ]
    layouts_for_report = {"uniform_U10": [30.0] * 10, **{key: list(value) for key, value in LAYOUTS.items()}}
    for index, item in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {item['layout_id']} | {item['mean_rmse']:.9g} | "
            f"{item['mean_max_abs']:.9g} | `{layouts_for_report[item['layout_id']]}` |"
        )
    lines.extend(["", "## Per-wavelength RMSE", "", "| layout | 60 | 120 | 180 | 240 | 300 |", "| :--- | ---: | ---: | ---: | ---: | ---: |"])
    for layout_id in layout_ids:
        rows = [item for item in results if item.layout_id == layout_id]
        lines.append(
            f"| {layout_id} | "
            + " | ".join(f"{item.rmse_vs_U30_serep_ridge:.9g}" for item in rows)
            + " |"
        )
    (OUTPUT_ROOT / "serep_ridge_nonuniform_design_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_workflow(*, force_hydro: bool, n_jobs: int) -> None:
    missing = [
        path
        for path in (
            STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
            STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
        )
        if not path.exists()
    ]
    missing.extend(
        path
        for path in (reference_response_path(wavelength_m) for wavelength_m in WAVELENGTHS_M)
        if not path.exists()
    )
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(str(path) for path in missing))

    results = add_uniform_baseline()
    geometry_paths: dict[str, Path] = {
        "uniform_U10": write_geometry_csv(
            "uniform_U10",
            build_hydro_config("uniform_U10", (30.0,) * 10, n_jobs=n_jobs),
        ),
        "uniform_U30_reference": write_geometry_csv(
            "uniform_U30_reference",
            build_hydro_config("uniform_U30_reference", (10.0,) * 30, n_jobs=n_jobs),
        ),
    }
    for layout_id, lengths in LAYOUTS.items():
        config = build_hydro_config(layout_id, lengths, n_jobs=n_jobs)
        ensure_hydrodynamics(config, force=force_hydro)
        geometry_paths[layout_id] = write_geometry_csv(layout_id, config)
        results.extend(solve_layout(layout_id, config))

    layout_ids = ["uniform_U10", *LAYOUTS.keys()]
    response_panel = plot_response_panel(layout_ids, results)
    error_summary = plot_error_summary(layout_ids, results)
    write_summary(
        results=results,
        geometry_paths=geometry_paths,
        response_panel=response_panel,
        error_summary=error_summary,
    )
    print(f"report={OUTPUT_ROOT / 'serep_ridge_nonuniform_design_report.md'}")
    print(f"response_panel={response_panel}")
    print(f"error_summary={error_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate hydrodynamic NC files.")
    parser.add_argument("--n-jobs", type=int, default=CAPYTAINE_N_JOBS, help="Capytaine worker count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_workflow(force_hydro=args.force_hydro, n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
