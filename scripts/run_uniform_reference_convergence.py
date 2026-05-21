"""Run U5/U10/U15/U30 uniform-module heave convergence comparisons.

The workflow keeps the physical body and structural FE matrices fixed, then
changes only the number of 1D uniform hydrodynamic modules along the length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import math
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
    omega_values_from_wavelengths,
    run_array_hydrodynamics,
)
from offshore_energy_sim.response.retained_dofs import retained_node_dof_series  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


LENGTH_M = 300.0
WIDTH_M = 60.0
HEIGHT_M = 2.0
DRAFT_M = 0.5
WATER_DEPTH_M = 58.5
RHO = 1000.0
G = 9.81
MESH_SIZE_M = 2.0
MODULE_COUNTS = (5, 10, 15, 30)
COMPARISON_PAIRS = tuple(zip(MODULE_COUNTS[:-1], MODULE_COUNTS[1:]))
WAVELENGTHS_M = (60, 120, 180, 240, 300)
CAPYTAINE_N_JOBS = min(16, max(1, (os.cpu_count() or 1) - 2))

TOTAL_NODES = 793
FULL_DOFS_PER_NODE = 6
RETAINED_DOFS_PER_NODE = 5
HYDRO_DOF_TO_REMOVE_ZERO_BASED = 5
REMOVED_FULL_DOFS_ZERO_BASED = (5,)
HYDRO_NODE_REVERSE_BY_WAVELENGTH: dict[int, bool] = {}

STRUCTURAL_DX_M = 5.0
STRUCTURAL_DY_M = 5.0
STRUCTURAL_NODES_PER_X = int(round(LENGTH_M / STRUCTURAL_DX_M)) + 1
STRUCTURAL_CENTERLINE_Y_INDEX = int(round((WIDTH_M / 2.0) / STRUCTURAL_DY_M))
STRUCTURAL_CENTERLINE_Y_M = WIDTH_M / 2.0

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "uniform_reference_convergence_U5_U10_U15_U30_heave"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
HYDRO_DIR = OUTPUT_ROOT / "hydro"
RESPONSE_DIR = OUTPUT_ROOT / "responses"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LEGACY_DM_FEM_ROOT = Path(r"E:\phd\Code\DM-FEM2D")
STRUCTURE_DIR = LEGACY_DM_FEM_ROOT / "StructureData"


def configure_output_paths(output_root: Path, *, hydro_dir: Path | None = None) -> None:
    """Configure output folders while allowing hydrodynamic NC reuse."""

    global OUTPUT_ROOT, HYDRO_DIR, RESPONSE_DIR, FIGURE_DIR
    OUTPUT_ROOT = output_root
    HYDRO_DIR = output_root / "hydro" if hydro_dir is None else hydro_dir
    RESPONSE_DIR = output_root / "responses"
    FIGURE_DIR = output_root / "figures"


@dataclass(frozen=True)
class ModuleGeometryRow:
    case_id: str
    module_id: int
    module_count: int
    module_length_m: float
    x_start_m: float
    x_end_m: float
    center_x_m: float
    hydrodynamic_center_x_m: float
    module_width_m: float
    module_height_m: float
    selected_node_id: int
    selected_node_x_m: float
    selected_node_y_m: float
    abs_error_m: float


@dataclass(frozen=True)
class ResponseCurve:
    x_over_l: np.ndarray
    values: np.ndarray


def module_length(module_count: int) -> float:
    return LENGTH_M / float(module_count)


def build_hydro_config(module_count: int, *, n_jobs: int) -> ArrayHydrodynamicsConfig:
    length_m = module_length(module_count)
    case_id = f"U{module_count}"
    return ArrayHydrodynamicsConfig(
        module=RectangularModuleSpec(
            length_m=length_m,
            width_m=WIDTH_M,
            height_m=HEIGHT_M,
            draft_m=DRAFT_M,
            mesh_size_m=MESH_SIZE_M,
            vertical_mesh_size_m=MESH_SIZE_M,
        ),
        layout=ArrayLayoutSpec(
            rows=1,
            columns=module_count,
            spacing_x_m=length_m,
            spacing_y_m=WIDTH_M,
            division_mode="uniform",
            total_length_m=LENGTH_M,
        ),
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, G),
        output_path=HYDRO_DIR
        / f"uniform_{case_id}_D0p5_rho1000_wl60_300_mesh2.nc",
        wave_directions_rad=(0.0,),
        water_depth_m=WATER_DEPTH_M,
        rho=RHO,
        g=G,
        n_jobs=n_jobs,
        compute_rao=False,
    )


def structural_paths() -> StructuralMatrixPaths:
    return StructuralMatrixPaths(
        mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
    )


def nearest_centerline_node(x_m: float) -> tuple[int, float, float, int]:
    """Return nearest 5 m-grid centerline node using half-up tie breaking."""

    x_index = int(math.floor(x_m / STRUCTURAL_DX_M + 0.5))
    x_index = min(max(x_index, 0), STRUCTURAL_NODES_PER_X - 1)
    selected_x = x_index * STRUCTURAL_DX_M
    node_x_index = STRUCTURAL_NODES_PER_X - 1 - x_index
    node_id = STRUCTURAL_CENTERLINE_Y_INDEX * STRUCTURAL_NODES_PER_X + node_x_index + 1
    return node_id, selected_x, STRUCTURAL_CENTERLINE_Y_M, x_index


def module_geometry_rows(module_count: int) -> list[ModuleGeometryRow]:
    length_m = module_length(module_count)
    hydrodynamic_x0 = -0.5 * (module_count - 1) * length_m
    rows: list[ModuleGeometryRow] = []
    for index in range(module_count):
        x_start = index * length_m
        x_end = x_start + length_m
        center_x = 0.5 * (x_start + x_end)
        hydro_center_x = hydrodynamic_x0 + index * length_m
        node_id, node_x, node_y, _ = nearest_centerline_node(center_x)
        rows.append(
            ModuleGeometryRow(
                case_id=f"U{module_count}",
                module_id=index + 1,
                module_count=module_count,
                module_length_m=length_m,
                x_start_m=x_start,
                x_end_m=x_end,
                center_x_m=center_x,
                hydrodynamic_center_x_m=hydro_center_x,
                module_width_m=WIDTH_M,
                module_height_m=HEIGHT_M,
                selected_node_id=node_id,
                selected_node_x_m=node_x,
                selected_node_y_m=node_y,
                abs_error_m=abs(node_x - center_x),
            )
        )
    return rows


def write_module_geometry_csv(module_count: int, rows: list[ModuleGeometryRow]) -> Path:
    path = OUTPUT_ROOT / f"U{module_count}_module_geometry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ModuleGeometryRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return path


def ensure_hydrodynamics(config: ArrayHydrodynamicsConfig, *, force: bool) -> None:
    if config.output_path.exists() and not force:
        return
    logs: list[str] = []
    start = time.perf_counter()
    run_array_hydrodynamics(config, log=logs.append)
    elapsed = time.perf_counter() - start
    config.output_path.with_suffix(".generation_log.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, indent=2),
        encoding="utf-8",
    )


def rodm_case(
    *,
    module_count: int,
    wavelength_index: int,
    hydro_path: Path,
    master_nodes_one_based: tuple[int, ...],
    structural_reduction_method: str = "serep",
    preserve_master_order: bool = False,
    robust_serep_mode_multiplier: float = 3.0,
    robust_serep_rcond: float = 1.0e-12,
    serep_ridge_relative_lambda: float = 1.0e-16,
) -> RodmFrequencyCase:
    wavelength_m = WAVELENGTHS_M[wavelength_index]
    return RodmFrequencyCase(
        case_id=f"U{module_count}_{wavelength_m}m",
        total_nodes=TOTAL_NODES,
        full_dofs_per_node=FULL_DOFS_PER_NODE,
        retained_dofs_per_node=RETAINED_DOFS_PER_NODE,
        removed_full_dofs_zero_based=REMOVED_FULL_DOFS_ZERO_BASED,
        master_node_rule=MasterNodeRule(
            first_node=master_nodes_one_based[0],
            node_interval=1,
            count=len(master_nodes_one_based),
        ),
        master_nodes_one_based=master_nodes_one_based,
        hydrodynamic_dataset=hydro_path,
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=module_count,
        hydrodynamic_dof_to_remove_zero_based=HYDRO_DOF_TO_REMOVE_ZERO_BASED,
        structural_reduction_method=structural_reduction_method,
        preserve_master_order=preserve_master_order,
        robust_serep_mode_multiplier=robust_serep_mode_multiplier,
        robust_serep_rcond=robust_serep_rcond,
        serep_ridge_relative_lambda=serep_ridge_relative_lambda,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(
            wavelength_m,
            False,
        ),
    )


def solve_and_save(case: RodmFrequencyCase, output_path: Path) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = solve_rodm_frequency_case(case).global_displacement
    np.save(output_path, response)
    return response


def extract_centerline_curve(response: np.ndarray, *, dof_index_zero_based: int) -> ResponseCurve:
    values_complex = retained_node_dof_series(
        response,
        start_node_one_based=367,
        stop_node_one_based=427,
        retained_dofs_per_node=RETAINED_DOFS_PER_NODE,
        dof_index_zero_based=dof_index_zero_based,
        column=0,
    )
    values = np.abs(values_complex)
    return ResponseCurve(x_over_l=np.linspace(0.0, 1.0, values.size), values=values)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(right) - np.asarray(left)
    return float(np.sqrt(np.mean(delta * delta)))


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(right) - np.asarray(left))))


def roughness(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(np.asarray(values), 2))))


def write_summary_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_ROOT / "reference_convergence_summary.csv"
    fieldnames = [
        "wavelength_m",
        "omega_rad_s",
    ]
    fieldnames.extend(f"rmse_U{left}_U{right}" for left, right in COMPARISON_PAIRS)
    fieldnames.extend(f"max_abs_U{left}_U{right}" for left, right in COMPARISON_PAIRS)
    fieldnames.extend(f"roughness_U{count}" for count in MODULE_COUNTS)
    fieldnames.extend(f"U{count}_response_path" for count in MODULE_COUNTS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_response_panel(
    curves: dict[tuple[int, int], ResponseCurve],
    output_path: Path,
    *,
    structural_reduction_method: str,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        5: {"color": "#9467bd", "linestyle": ":", "label": "U5"},
        10: {"color": "#1f77b4", "linestyle": "-", "label": "U10"},
        15: {"color": "#d62728", "linestyle": "--", "label": "U15"},
        30: {"color": "#2ca02c", "linestyle": "-.", "label": "U30"},
    }
    fig, axes = plt.subplots(len(WAVELENGTHS_M), 1, figsize=(9.8, 15.5), sharex=True)
    for ax, wavelength_m in zip(np.atleast_1d(axes), WAVELENGTHS_M):
        for module_count in MODULE_COUNTS:
            curve = curves[(module_count, wavelength_m)]
            ax.plot(
                curve.x_over_l,
                curve.values,
                linewidth=1.7,
                **styles[module_count],
            )
        ax.set_title(f"Heave, wavelength {wavelength_m} m")
        ax.set_ylabel("Heave RAO (m/m)")
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    np.atleast_1d(axes)[-1].set_xlabel("x/L")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Uniform Module Heave Convergence: "
        f"U5 vs U10 vs U15 vs U30 ({structural_reduction_method})",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_error_summary(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    structural_reduction_method: str,
) -> None:
    import matplotlib.pyplot as plt

    wavelengths = np.asarray(WAVELENGTHS_M, dtype=float)
    bar_width = 8.0
    offsets = np.linspace(-bar_width, bar_width, len(COMPARISON_PAIRS))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("rmse", "max_abs"),
        ("Heave RMSE", "Heave Max Abs Difference"),
    ):
        for offset, (left, right) in zip(offsets, COMPARISON_PAIRS):
            values = [float(row[f"{metric}_U{left}_U{right}"]) for row in rows]
            ax.bar(
                wavelengths + offset,
                values,
                width=bar_width * 0.82,
                label=f"U{left} vs U{right}",
            )
        ax.set_title(title)
        ax.set_ylabel("difference")
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    for ax in axes:
        ax.set_xlabel("wavelength (m)")
        ax.set_xticks(wavelengths)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Uniform Module Heave Convergence Error Summary ({structural_reduction_method})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def compute_serep_diagnostics(
    master_nodes: dict[int, tuple[int, ...]],
) -> dict[str, dict[str, object]]:
    """Measure conditioning of the SEREP master modal block for each case."""

    from scipy.linalg import eigh

    from offshore_energy_sim.reduction import (
        reduce_matrix_dofs,
        separate_master_slave_dofs,
        transform_mass_matrix,
    )
    from offshore_energy_sim.structure.matrix_io import read_abaqus_matrix_dense

    mass_full = read_abaqus_matrix_dense(
        STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        dofs_per_node=FULL_DOFS_PER_NODE,
    )
    stiffness_full = read_abaqus_matrix_dense(
        STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
        dofs_per_node=FULL_DOFS_PER_NODE,
    )
    mass = transform_mass_matrix(
        reduce_matrix_dofs(mass_full, TOTAL_NODES, REMOVED_FULL_DOFS_ZERO_BASED),
        beta=0.0,
    )
    stiffness = reduce_matrix_dofs(
        stiffness_full,
        TOTAL_NODES,
        REMOVED_FULL_DOFS_ZERO_BASED,
    )

    diagnostics: dict[str, dict[str, object]] = {}
    for module_count, nodes in master_nodes.items():
        _, slave_dofs = separate_master_slave_dofs(
            TOTAL_NODES,
            nodes,
            dofs_per_node=RETAINED_DOFS_PER_NODE,
        )
        slave_dofs = np.sort(slave_dofs)
        master_dofs = np.setdiff1d(np.arange(stiffness.shape[0]), slave_dofs)
        order = np.concatenate([master_dofs, slave_dofs])
        reordered_stiffness = stiffness[np.ix_(order, order)]
        reordered_mass = mass[np.ix_(order, order)]
        eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
        for mode_index in range(eigenvectors.shape[1]):
            max_value = np.max(np.abs(eigenvectors[:, mode_index]))
            eigenvectors[:, mode_index] /= max_value

        master_size = RETAINED_DOFS_PER_NODE * len(nodes)
        master_modal_block = eigenvectors[:master_size, :master_size]
        singular_values = np.linalg.svd(master_modal_block, compute_uv=False)
        diagnostics[f"U{module_count}"] = {
            "master_node_count": len(nodes),
            "master_dof_count": master_size,
            "modal_block_condition": float(singular_values[0] / singular_values[-1]),
            "modal_block_smin": float(singular_values[-1]),
            "modal_block_smax": float(singular_values[0]),
            "first_five_eigenvalues": [float(value) for value in eigenvalues[:5]],
            "last_retained_eigenvalue": float(eigenvalues[master_size - 1]),
        }
    return diagnostics


def run_workflow(
    *,
    force_hydro: bool,
    n_jobs: int,
    structural_reduction_method: str = "serep",
    preserve_master_order: bool = False,
    robust_serep_mode_multiplier: float = 3.0,
    robust_serep_rcond: float = 1.0e-12,
    serep_ridge_relative_lambda: float = 1.0e-16,
) -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    hydro_configs = {count: build_hydro_config(count, n_jobs=n_jobs) for count in MODULE_COUNTS}
    geometry_by_count = {count: module_geometry_rows(count) for count in MODULE_COUNTS}
    geometry_paths = {
        count: write_module_geometry_csv(count, rows)
        for count, rows in geometry_by_count.items()
    }
    master_nodes = {
        count: tuple(row.selected_node_id for row in rows)
        for count, rows in geometry_by_count.items()
    }

    for config in hydro_configs.values():
        ensure_hydrodynamics(config, force=force_hydro)

    responses: dict[tuple[int, int], np.ndarray] = {}
    response_paths: dict[tuple[int, int], Path] = {}
    omegas = tuple(float(value) for value in hydro_configs[10].omegas_rad_s)
    for module_count in MODULE_COUNTS:
        for wavelength_index, wavelength_m in enumerate(WAVELENGTHS_M):
            response_path = (
                RESPONSE_DIR
                / f"U{module_count}"
                / f"uniform_U{module_count}_wavelength_{wavelength_m}m_response.npy"
            )
            case = rodm_case(
                module_count=module_count,
                wavelength_index=wavelength_index,
                hydro_path=hydro_configs[module_count].output_path,
                master_nodes_one_based=master_nodes[module_count],
                structural_reduction_method=structural_reduction_method,
                preserve_master_order=preserve_master_order,
                robust_serep_mode_multiplier=robust_serep_mode_multiplier,
                robust_serep_rcond=robust_serep_rcond,
                serep_ridge_relative_lambda=serep_ridge_relative_lambda,
            )
            responses[(module_count, wavelength_m)] = solve_and_save(case, response_path)
            response_paths[(module_count, wavelength_m)] = response_path

    curves: dict[tuple[int, int], ResponseCurve] = {}
    for module_count in MODULE_COUNTS:
        for wavelength_m in WAVELENGTHS_M:
            response = responses[(module_count, wavelength_m)]
            curves[(module_count, wavelength_m)] = extract_centerline_curve(
                response,
                dof_index_zero_based=2,
            )

    summary_rows: list[dict[str, object]] = []
    for wavelength_index, wavelength_m in enumerate(WAVELENGTHS_M):
        row: dict[str, object] = {
            "wavelength_m": wavelength_m,
            "omega_rad_s": omegas[wavelength_index],
        }
        for left, right in COMPARISON_PAIRS:
            left_values = curves[(left, wavelength_m)].values
            right_values = curves[(right, wavelength_m)].values
            row[f"rmse_U{left}_U{right}"] = rmse(left_values, right_values)
            row[f"max_abs_U{left}_U{right}"] = max_abs_delta(left_values, right_values)
        for module_count in MODULE_COUNTS:
            row[f"roughness_U{module_count}"] = roughness(
                curves[(module_count, wavelength_m)].values
            )
            row[f"U{module_count}_response_path"] = str(
                response_paths[(module_count, wavelength_m)]
            )
        summary_rows.append(row)

    summary_csv = write_summary_csv(summary_rows)
    response_panel = FIGURE_DIR / "U5_vs_U10_vs_U15_vs_U30_heave_response_panel.png"
    error_summary = FIGURE_DIR / "U5_U10_U15_U30_heave_error_summary.png"
    plot_response_panel(
        curves,
        response_panel,
        structural_reduction_method=structural_reduction_method,
    )
    plot_error_summary(
        summary_rows,
        error_summary,
        structural_reduction_method=structural_reduction_method,
    )
    serep_diagnostics = compute_serep_diagnostics(master_nodes)
    diagnostics_path = OUTPUT_ROOT / "reference_convergence_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps({"serep": serep_diagnostics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "parameters": {
            "length_m": LENGTH_M,
            "width_m": WIDTH_M,
            "height_m": HEIGHT_M,
            "draft_m": DRAFT_M,
            "water_depth_m": WATER_DEPTH_M,
            "rho": RHO,
            "g": G,
            "mesh_size_m": MESH_SIZE_M,
            "module_counts": list(MODULE_COUNTS),
            "wavelengths_m": list(WAVELENGTHS_M),
            "capytaine_n_jobs": n_jobs,
            "hydro_reverse_by_wavelength": HYDRO_NODE_REVERSE_BY_WAVELENGTH,
            "structural_reduction_method": structural_reduction_method,
            "preserve_master_order": preserve_master_order,
            "robust_serep_mode_multiplier": robust_serep_mode_multiplier,
            "robust_serep_rcond": robust_serep_rcond,
            "serep_ridge_relative_lambda": serep_ridge_relative_lambda,
        },
        "hydrodynamic_files": {
            f"U{count}": str(config.output_path)
            for count, config in hydro_configs.items()
        },
        "module_geometry_csv": {
            f"U{count}": str(path)
            for count, path in geometry_paths.items()
        },
        "summary_csv": str(summary_csv),
        "diagnostics_json": str(diagnostics_path),
        "figures": {
            "response_panel": str(response_panel),
            "error_summary": str(error_summary),
        },
        "master_nodes": {
            f"U{count}": list(nodes)
            for count, nodes in master_nodes.items()
        },
        "control_points": {
            f"U{count}": [
                {
                    "module_id": row.module_id,
                    "center_x_m": row.center_x_m,
                    "selected_node_id": row.selected_node_id,
                    "selected_node_x_m": row.selected_node_x_m,
                    "abs_error_m": row.abs_error_m,
                }
                for row in rows
            ]
            for count, rows in geometry_by_count.items()
        },
        "serep_diagnostics": serep_diagnostics,
    }
    (OUTPUT_ROOT / "reference_convergence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(manifest, summary_rows)
    return manifest


def write_markdown_report(manifest: dict[str, object], rows: list[dict[str, object]]) -> None:
    metric_header = ["| wavelength m |"]
    metric_separator = ["| ---: |"]
    for left, right in COMPARISON_PAIRS:
        metric_header.append(f" RMSE U{left}-U{right} |")
        metric_separator.append(" ---: |")
    for left, right in COMPARISON_PAIRS:
        metric_header.append(f" max U{left}-U{right} |")
        metric_separator.append(" ---: |")
    for count in MODULE_COUNTS:
        metric_header.append(f" rough U{count} |")
        metric_separator.append(" ---: |")

    lines = [
        "# Uniform Module Reference Convergence",
        "",
        "Uniform 1D hydrodynamic module layouts were compared for heave only: U5, U10, U15, and U30.",
        "",
        f"- structural reduction method: `{manifest['parameters']['structural_reduction_method']}`",
        f"- preserve master order: `{manifest['parameters']['preserve_master_order']}`",
        "",
        f"- summary CSV: `{manifest['summary_csv']}`",
        f"- diagnostics JSON: `{manifest['diagnostics_json']}`",
        f"- response panel: `{manifest['figures']['response_panel']}`",
        f"- error summary: `{manifest['figures']['error_summary']}`",
        "",
        "## Module Geometry",
        "",
    ]
    for case_id, path in manifest["module_geometry_csv"].items():
        lines.append(f"- {case_id}: `{path}`")
    lines.extend(
        [
            "",
            "## Main Control Points",
            "",
            "| case | module length m | selected node ids | max center-node error m |",
            "| :--- | ---: | :--- | ---: |",
        ]
    )
    for count in MODULE_COUNTS:
        case_id = f"U{count}"
        control_points = manifest["control_points"][case_id]
        node_ids = [str(item["selected_node_id"]) for item in control_points]
        max_error = max(float(item["abs_error_m"]) for item in control_points)
        lines.append(
            f"| {case_id} | {module_length(count):.9g} | "
            f"`[{', '.join(node_ids)}]` | {max_error:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Convergence Metrics",
            "",
            "".join(metric_header),
            "".join(metric_separator),
        ]
    )
    for row in rows:
        values = [f"| {row['wavelength_m']} |"]
        for left, right in COMPARISON_PAIRS:
            values.append(f" {float(row[f'rmse_U{left}_U{right}']):.9g} |")
        for left, right in COMPARISON_PAIRS:
            values.append(f" {float(row[f'max_abs_U{left}_U{right}']):.9g} |")
        for count in MODULE_COUNTS:
            values.append(f" {float(row[f'roughness_U{count}']):.9g} |")
        lines.append("".join(values))
    lines.extend(
        [
            "",
            "## Legacy SEREP Conditioning",
            "",
            "| case | master nodes | master DOFs | modal block condition | smallest singular value |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_id, values in manifest["serep_diagnostics"].items():
        lines.append(
            f"| {case_id} | {values['master_node_count']} | {values['master_dof_count']} | "
            f"{float(values['modal_block_condition']):.9g} | "
            f"{float(values['modal_block_smin']):.9g} |"
        )
    lines.append("")
    (OUTPUT_ROOT / "reference_convergence_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate hydrodynamic NC files.")
    parser.add_argument("--n-jobs", type=int, default=CAPYTAINE_N_JOBS, help="Capytaine parallel worker count.")
    parser.add_argument(
        "--structural-reduction-method",
        choices=("serep", "guyan_static", "serep_robust", "serep_ridge"),
        default="serep",
        help="Structural reduction method used by the RODM solve.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override result folder. Defaults to a method-specific folder.",
    )
    parser.add_argument(
        "--hydro-dir",
        type=Path,
        default=None,
        help="Hydrodynamic NC folder. Defaults to reusing the legacy uniform hydro folder.",
    )
    parser.add_argument(
        "--preserve-master-order",
        action="store_true",
        help="Keep explicit master-node order in the structural reduced coordinates.",
    )
    parser.add_argument(
        "--robust-serep-mode-multiplier",
        type=float,
        default=3.0,
        help="Mode-count multiplier for robust SEREP.",
    )
    parser.add_argument(
        "--robust-serep-rcond",
        type=float,
        default=1.0e-12,
        help="SVD pseudoinverse cutoff for robust SEREP.",
    )
    parser.add_argument(
        "--serep-ridge-relative-lambda",
        type=float,
        default=1.0e-16,
        help="Relative Tikhonov regularization for ridge SEREP.",
    )
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
    if missing:
        raise FileNotFoundError("Missing structural inputs: " + ", ".join(str(path) for path in missing))

    if args.output_root is not None:
        output_root = args.output_root
    elif args.structural_reduction_method == "serep":
        output_root = DEFAULT_OUTPUT_ROOT
    else:
        order_suffix = "_ordered" if args.preserve_master_order else ""
        output_root = (
            REPO_ROOT
            / "results"
            / (
                "uniform_reference_convergence_U5_U10_U15_U30_heave_"
                f"{args.structural_reduction_method}_forward_order{order_suffix}"
            )
        )

    hydro_dir = args.hydro_dir
    if hydro_dir is None and args.structural_reduction_method != "serep":
        hydro_dir = DEFAULT_OUTPUT_ROOT / "hydro"
    configure_output_paths(output_root, hydro_dir=hydro_dir)

    manifest = run_workflow(
        force_hydro=args.force_hydro,
        n_jobs=args.n_jobs,
        structural_reduction_method=args.structural_reduction_method,
        preserve_master_order=args.preserve_master_order,
        robust_serep_mode_multiplier=args.robust_serep_mode_multiplier,
        robust_serep_rcond=args.robust_serep_rcond,
        serep_ridge_relative_lambda=args.serep_ridge_relative_lambda,
    )
    print(f"summary_csv={manifest['summary_csv']}")
    print(f"report={OUTPUT_ROOT / 'reference_convergence_report.md'}")
    print(f"response_panel={manifest['figures']['response_panel']}")
    print(f"error_summary={manifest['figures']['error_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
