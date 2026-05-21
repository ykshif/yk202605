"""Compare node-aligned non-uniform U10 RODM against the U30/Guyan reference.

The hydrodynamic mesh/data generation is non-uniform along length only. The
structural reduction uses the repaired ``guyan_static`` method so that the test
isolates hydrodynamic module discretization rather than the legacy SEREP
conditioning issue.
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

from offshore_energy_sim.core import (  # noqa: E402
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)
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
MODULE_LENGTHS_M = (20.0, 20.0, 30.0, 40.0, 40.0, 40.0, 40.0, 30.0, 20.0, 20.0)
CAPYTAINE_N_JOBS = min(16, max(1, (os.cpu_count() or 1) - 2))

OUTPUT_ROOT = REPO_ROOT / "results" / "nonuniform_U10_vs_U30_guyan_forward_ordered_reference"
HYDRO_OUTPUT = (
    OUTPUT_ROOT
    / "hydro"
    / "edge_refined_nonuniform_U10_D0p5_rho1000_wl60_300_mesh2.nc"
)
REFERENCE_ROOT = (
    REPO_ROOT
    / "results"
    / "uniform_reference_convergence_U5_U10_U15_U30_heave_guyan_static_forward_order_ordered"
)
DM_FEM_ROOT = Path(r"E:\phd\Code\DM-FEM2D")
STRUCTURE_DIR = DM_FEM_ROOT / "StructureData"
STRUCTURAL_REDUCTION_METHOD = "guyan_static"
ORIENTATION_VARIANTS = {
    "hydro_order_forward": False,
    "hydro_order_reversed": True,
}


@dataclass(frozen=True)
class ComparisonRow:
    wavelength_m: int
    omega_rad_s: float
    variant: str
    reverse_hydrodynamic_node_order: bool
    heave_rmse_vs_U30_guyan: float
    heave_max_abs_vs_U30_guyan: float
    nonuniform_heave_min: float
    nonuniform_heave_max: float
    U30_reference_heave_min: float
    U30_reference_heave_max: float
    response_path: Path
    reference_response_path: Path
    figure_path: Path


def build_nonuniform_hydro_config(*, n_jobs: int) -> ArrayHydrodynamicsConfig:
    module = RectangularModuleSpec(
        length_m=30.0,
        width_m=WIDTH_M,
        height_m=HEIGHT_M,
        draft_m=DRAFT_M,
        mesh_size_m=MESH_SIZE_M,
        vertical_mesh_size_m=MESH_SIZE_M,
    )
    layout = ArrayLayoutSpec(
        rows=1,
        columns=len(MODULE_LENGTHS_M),
        spacing_x_m=30.0,
        spacing_y_m=WIDTH_M,
        division_mode="custom",
        total_length_m=LENGTH_M,
        module_lengths_x_m=MODULE_LENGTHS_M,
    )
    return ArrayHydrodynamicsConfig(
        module=module,
        layout=layout,
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, G),
        output_path=HYDRO_OUTPUT,
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
    log_path = config.output_path.with_suffix(".generation_log.json")
    log_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def structural_paths() -> StructuralMatrixPaths:
    return StructuralMatrixPaths(
        mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
    )


def build_nonuniform_case(
    *,
    wavelength_index: int,
    master_nodes: tuple[int, ...],
    reverse_hydrodynamic_node_order: bool,
) -> RodmFrequencyCase:
    wavelength_m = WAVELENGTHS_M[wavelength_index]
    return RodmFrequencyCase(
        case_id=f"nonuniform_U10_{wavelength_m}m_{STRUCTURAL_REDUCTION_METHOD}",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=master_nodes[0], node_interval=1, count=len(master_nodes)),
        master_nodes_one_based=master_nodes,
        hydrodynamic_dataset=HYDRO_OUTPUT,
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=len(master_nodes),
        hydrodynamic_dof_to_remove_zero_based=5,
        structural_reduction_method=STRUCTURAL_REDUCTION_METHOD,
        preserve_master_order=True,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=reverse_hydrodynamic_node_order,
    )


def reference_response_path(wavelength_m: int) -> Path:
    return (
        REFERENCE_ROOT
        / "responses"
        / "U30"
        / f"uniform_U30_wavelength_{wavelength_m}m_response.npy"
    )


def uniform_response_path(module_count: int, wavelength_m: int) -> Path:
    return (
        REFERENCE_ROOT
        / "responses"
        / f"U{module_count}"
        / f"uniform_U{module_count}_wavelength_{wavelength_m}m_response.npy"
    )


def solve_and_save(case: RodmFrequencyCase, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = solve_rodm_frequency_case(case).global_displacement
    np.save(path, response)
    return response


def write_module_geometry_csv(config: ArrayHydrodynamicsConfig) -> Path:
    rows = []
    mappings = module_structural_node_mappings(config)
    boundaries = config.layout.x_boundaries(config.module.length_m)
    geometries = config.layout.module_geometries(config.module.length_m)
    for index, (geometry, mapping) in enumerate(zip(geometries, mappings), start=1):
        rows.append(
            {
                "module_id": index,
                "module_length_m": geometry.length_m,
                "x_start_m": boundaries[index - 1],
                "x_end_m": boundaries[index],
                "center_x_m": geometry.x_m,
                "width_m": WIDTH_M,
                "height_m": HEIGHT_M,
                "selected_node_id": mapping["fem_node_one_based"],
                "selected_node_x_m": mapping["x_m"],
                "selected_node_y_m": mapping["y_m"],
                "abs_error_m": abs(float(mapping["x_m"]) - geometry.x_m),
            }
        )
    path = OUTPUT_ROOT / "nonuniform_U10_module_geometry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_wavelength_comparison(
    *,
    wavelength_m: int,
    reference_response: np.ndarray,
    nonuniform_responses: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x_ref, heave_ref = extract_centerline_heave(reference_response)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(x_ref, heave_ref, color="#111111", linewidth=2.1, label="U30 Guyan reference")
    styles = {
        "hydro_order_forward": {"color": "#1f77b4", "linestyle": "--"},
        "hydro_order_reversed": {"color": "#d62728", "linestyle": "-."},
    }
    for variant, response in nonuniform_responses.items():
        x_non, heave_non = extract_centerline_heave(response)
        ax.plot(
            x_non,
            heave_non,
            linewidth=1.7,
            label=f"Non-uniform U10 {variant}",
            **styles[variant],
        )
    ax.set_title(f"Non-uniform U10 vs U30 Guyan reference, wavelength {wavelength_m} m")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_summary(
    rows: list[ComparisonRow],
    baseline_rows: list[dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavelengths = np.asarray(WAVELENGTHS_M, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    width = 10.0
    for offset, variant in zip((-0.5 * width, 0.5 * width), ORIENTATION_VARIANTS):
        variant_rows = [row for row in rows if row.variant == variant]
        axes[0].bar(
            wavelengths + offset,
            [row.heave_rmse_vs_U30_guyan for row in variant_rows],
            width=0.82 * width,
            label=variant,
        )
        axes[1].bar(
            wavelengths + offset,
            [row.heave_max_abs_vs_U30_guyan for row in variant_rows],
            width=0.82 * width,
            label=variant,
        )
    baseline_by_wavelength = {int(row["wavelength_m"]): row for row in baseline_rows}
    axes[0].plot(
        wavelengths,
        [baseline_by_wavelength[int(value)]["heave_rmse_vs_U30_guyan"] for value in wavelengths],
        color="#111111",
        marker="o",
        linewidth=1.6,
        label="uniform_U10",
    )
    axes[1].plot(
        wavelengths,
        [baseline_by_wavelength[int(value)]["heave_max_abs_vs_U30_guyan"] for value in wavelengths],
        color="#111111",
        marker="o",
        linewidth=1.6,
        label="uniform_U10",
    )
    axes[0].set_title("Heave RMSE")
    axes[1].set_title("Heave Max Abs Difference")
    for axis in axes:
        axis.set_xlabel("wavelength (m)")
        axis.set_xticks(wavelengths)
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].set_ylabel("difference")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Non-uniform U10 vs U30/Guyan Reference", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(
    *,
    config: ArrayHydrodynamicsConfig,
    rows: list[ComparisonRow],
    baseline_rows: list[dict[str, float]],
    geometry_csv: Path,
    summary_figure: Path,
) -> None:
    csv_path = OUTPUT_ROOT / "nonuniform_vs_U30_guyan_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ComparisonRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            for key in ("response_path", "reference_response_path", "figure_path"):
                data[key] = str(data[key])
            writer.writerow(data)

    baseline_csv_path = OUTPUT_ROOT / "uniform_U10_vs_U30_guyan_baseline.csv"
    with baseline_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)

    mappings = module_structural_node_mappings(config)
    summary = {
        "parameters": {
            "length_m": LENGTH_M,
            "width_m": WIDTH_M,
            "height_m": HEIGHT_M,
            "draft_m": DRAFT_M,
            "water_depth_m": WATER_DEPTH_M,
            "rho": RHO,
            "mesh_size_m": MESH_SIZE_M,
            "wavelengths_m": list(WAVELENGTHS_M),
            "module_lengths_m": list(MODULE_LENGTHS_M),
            "structural_reduction_method": STRUCTURAL_REDUCTION_METHOD,
            "preserve_master_order": True,
        },
        "nonuniform_hydrodynamic_dataset": str(HYDRO_OUTPUT),
        "U30_reference_root": str(REFERENCE_ROOT),
        "module_geometry_csv": str(geometry_csv),
        "summary_csv": str(csv_path),
        "uniform_U10_baseline_csv": str(baseline_csv_path),
        "summary_figure": str(summary_figure),
        "module_centers_x_m": [item["hydrodynamic_x_m"] for item in mappings],
        "selected_node_ids": [item["fem_node_one_based"] for item in mappings],
        "results": [
            {
                **{
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in row.__dict__.items()
                }
            }
            for row in rows
        ],
        "uniform_U10_baseline": baseline_rows,
    }
    json_path = OUTPUT_ROOT / "nonuniform_vs_U30_guyan_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Non-uniform U10 vs U30/Guyan Reference",
        "",
        "The non-uniform modules are 1D along length only and every module spans the full 60 m width.",
        "",
        f"- structural reduction method: `{STRUCTURAL_REDUCTION_METHOD}`",
        "- preserve master order: `True`",
        f"- module lengths m: `{list(MODULE_LENGTHS_M)}`",
        f"- selected node ids: `{summary['selected_node_ids']}`",
        f"- hydrodynamic dataset: `{HYDRO_OUTPUT}`",
        f"- module geometry CSV: `{geometry_csv}`",
        f"- summary CSV: `{csv_path}`",
        f"- uniform U10 baseline CSV: `{baseline_csv_path}`",
        f"- summary figure: `{summary_figure}`",
        "",
        "| wavelength m | variant | RMSE vs U30/Guyan | max abs vs U30/Guyan | figure |",
        "| ---: | :--- | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.wavelength_m} | {row.variant} | "
            f"{row.heave_rmse_vs_U30_guyan:.9g} | "
            f"{row.heave_max_abs_vs_U30_guyan:.9g} | "
            f"`{row.figure_path}` |"
        )
    lines.extend(
        [
            "",
            "## Uniform U10 Baseline",
            "",
            "| wavelength m | RMSE U10 vs U30/Guyan | max abs U10 vs U30/Guyan |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {int(row['wavelength_m'])} | "
            f"{row['heave_rmse_vs_U30_guyan']:.9g} | "
            f"{row['heave_max_abs_vs_U30_guyan']:.9g} |"
        )
    report_path = OUTPUT_ROOT / "nonuniform_vs_U30_guyan_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_workflow(*, force_hydro: bool, n_jobs: int) -> dict[str, Path]:
    config = build_nonuniform_hydro_config(n_jobs=n_jobs)
    ensure_hydrodynamics(config, force=force_hydro)
    geometry_csv = write_module_geometry_csv(config)
    mappings = module_structural_node_mappings(config)
    master_nodes = tuple(int(item["fem_node_one_based"]) for item in mappings)
    omegas = tuple(float(value) for value in config.omegas_rad_s)

    rows: list[ComparisonRow] = []
    baseline_rows: list[dict[str, float]] = []
    for wavelength_index, wavelength_m in enumerate(WAVELENGTHS_M):
        ref_path = reference_response_path(wavelength_m)
        reference_response = np.load(ref_path)
        _, heave_reference = extract_centerline_heave(reference_response)
        uniform_u10_response = np.load(uniform_response_path(10, wavelength_m))
        _, heave_uniform_u10 = extract_centerline_heave(uniform_u10_response)
        uniform_delta = heave_uniform_u10 - heave_reference
        baseline_rows.append(
            {
                "wavelength_m": float(wavelength_m),
                "omega_rad_s": omegas[wavelength_index],
                "heave_rmse_vs_U30_guyan": float(np.sqrt(np.mean(uniform_delta**2))),
                "heave_max_abs_vs_U30_guyan": float(np.max(np.abs(uniform_delta))),
            }
        )
        responses_by_variant: dict[str, np.ndarray] = {}

        figure_path = OUTPUT_ROOT / "figures" / f"nonuniform_vs_U30_guyan_{wavelength_m}m.png"
        for variant, reverse in ORIENTATION_VARIANTS.items():
            response_path = (
                OUTPUT_ROOT
                / "responses"
                / variant
                / f"nonuniform_U10_{wavelength_m}m_response.npy"
            )
            case = build_nonuniform_case(
                wavelength_index=wavelength_index,
                master_nodes=master_nodes,
                reverse_hydrodynamic_node_order=reverse,
            )
            response = solve_and_save(case, response_path)
            responses_by_variant[variant] = response
            _, heave_nonuniform = extract_centerline_heave(response)
            delta = heave_nonuniform - heave_reference
            rows.append(
                ComparisonRow(
                    wavelength_m=wavelength_m,
                    omega_rad_s=omegas[wavelength_index],
                    variant=variant,
                    reverse_hydrodynamic_node_order=reverse,
                    heave_rmse_vs_U30_guyan=float(np.sqrt(np.mean(delta**2))),
                    heave_max_abs_vs_U30_guyan=float(np.max(np.abs(delta))),
                    nonuniform_heave_min=float(np.min(heave_nonuniform)),
                    nonuniform_heave_max=float(np.max(heave_nonuniform)),
                    U30_reference_heave_min=float(np.min(heave_reference)),
                    U30_reference_heave_max=float(np.max(heave_reference)),
                    response_path=response_path,
                    reference_response_path=ref_path,
                    figure_path=figure_path,
                )
            )
        plot_wavelength_comparison(
            wavelength_m=wavelength_m,
            reference_response=reference_response,
            nonuniform_responses=responses_by_variant,
            output_path=figure_path,
        )

    summary_figure = OUTPUT_ROOT / "figures" / "nonuniform_U10_vs_U30_guyan_error_summary.png"
    plot_summary(rows, baseline_rows, summary_figure)
    write_summary(
        config=config,
        rows=rows,
        baseline_rows=baseline_rows,
        geometry_csv=geometry_csv,
        summary_figure=summary_figure,
    )
    return {
        "summary_csv": OUTPUT_ROOT / "nonuniform_vs_U30_guyan_summary.csv",
        "report": OUTPUT_ROOT / "nonuniform_vs_U30_guyan_report.md",
        "summary_figure": summary_figure,
        "geometry_csv": geometry_csv,
        "hydro": HYDRO_OUTPUT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate the non-uniform hydrodynamic NC file.")
    parser.add_argument("--n-jobs", type=int, default=CAPYTAINE_N_JOBS, help="Capytaine worker count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    outputs = run_workflow(force_hydro=args.force_hydro, n_jobs=args.n_jobs)
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
