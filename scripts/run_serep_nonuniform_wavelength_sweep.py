"""Run an extended wavelength sweep for representative NU10 SEREP-ridge layouts.

This is the next-step experiment after the final five-wavelength validation.
It does not search new layouts. It freezes a small set of representative
layouts and scans more wavelengths to build an applicability map.
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_uniform_reference_convergence as uniform  # noqa: E402
import run_serep_nonuniform_design_study as base  # noqa: E402
from offshore_energy_sim.core import MasterNodeRule, RodmFrequencyCase  # noqa: E402
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
SEREP_RIDGE_RELATIVE_LAMBDA = 1.0e-16
DEFAULT_WAVELENGTHS_M = (60, 90, 120, 150, 180, 210, 240, 270, 300)
CAPYTAINE_N_JOBS = min(16, max(1, (os.cpu_count() or 1) - 2))

OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_wavelength_sweep"
HYDRO_DIR = OUTPUT_ROOT / "hydro"
RESPONSE_DIR = OUTPUT_ROOT / "responses"
GEOMETRY_DIR = OUTPUT_ROOT / "geometry"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_PATH = OUTPUT_ROOT / "serep_nonuniform_wavelength_sweep_report.md"


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    display_name: str
    category: str
    module_lengths_m: tuple[float, ...]

    @property
    def module_count(self) -> int:
        return len(self.module_lengths_m)

    @property
    def is_uniform(self) -> bool:
        return len(set(self.module_lengths_m)) == 1


LAYOUTS = (
    LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
    LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
    LayoutSpec("NU10_center_refined", "NU10 center-refined", "rule", (40, 40, 30, 20, 20, 20, 20, 30, 40, 40)),
    LayoutSpec("NU10_edge_mild", "NU10 edge-mild", "rule", (20, 30, 30, 40, 30, 30, 40, 30, 30, 20)),
    LayoutSpec("NU10_bow_refined", "NU10 bow-refined", "rule", (20, 20, 20, 30, 30, 30, 30, 40, 40, 40)),
    LayoutSpec("NU10_mean_best", "NU10 mean-best", "searched", (30, 30, 30, 30, 30, 30, 40, 30, 30, 20)),
)


COLORS = {
    "uniform_U10": "#1f77b4",
    "NU10_center_refined": "#ff7f0e",
    "NU10_edge_mild": "#9467bd",
    "NU10_bow_refined": "#d62728",
    "NU10_mean_best": "#2ca02c",
}


def selected_layouts(*, include_searched: bool) -> tuple[LayoutSpec, ...]:
    if include_searched:
        return LAYOUTS
    return tuple(item for item in LAYOUTS if item.category != "searched")


def parse_wavelengths(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_WAVELENGTHS_M
    values = tuple(int(float(value)) for value in text.replace(",", " ").split())
    if len(values) == 0:
        raise ValueError("at least one wavelength is required")
    return values


def output_suffix(wavelengths_m: tuple[int, ...]) -> str:
    return f"wl{min(wavelengths_m)}_{max(wavelengths_m)}_n{len(wavelengths_m)}_mesh2"


def hydro_config(layout: LayoutSpec, wavelengths_m: tuple[int, ...], *, n_jobs: int) -> ArrayHydrodynamicsConfig:
    suffix = output_suffix(wavelengths_m)
    if layout.is_uniform:
        length_m = layout.module_lengths_m[0]
        division_mode = "uniform"
        module_lengths = None
    else:
        length_m = 30.0
        division_mode = "custom"
        module_lengths = layout.module_lengths_m

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
            columns=layout.module_count,
            spacing_x_m=length_m,
            spacing_y_m=WIDTH_M,
            division_mode=division_mode,
            total_length_m=LENGTH_M,
            module_lengths_x_m=module_lengths,
        ),
        omegas_rad_s=omega_values_from_wavelengths(wavelengths_m, WATER_DEPTH_M, G),
        output_path=HYDRO_DIR / f"{layout.layout_id}_D0p5_rho1000_{suffix}.nc",
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
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    start = time.perf_counter()
    run_array_hydrodynamics(config, log=logs.append)
    elapsed = time.perf_counter() - start
    config.output_path.with_suffix(".generation_log.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def uniform_geometry_rows(layout: LayoutSpec) -> list[dict[str, object]]:
    module_count = layout.module_count
    rows = []
    for row in uniform.module_geometry_rows(module_count):
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_id": row.module_id,
                "module_length_m": row.module_length_m,
                "x_start_m": row.x_start_m,
                "x_end_m": row.x_end_m,
                "center_x_m": row.center_x_m,
                "width_m": WIDTH_M,
                "height_m": HEIGHT_M,
                "selected_node_id": row.selected_node_id,
                "selected_node_x_m": row.selected_node_x_m,
                "selected_node_y_m": row.selected_node_y_m,
                "abs_error_m": row.abs_error_m,
            }
        )
    return rows


def nonuniform_geometry_rows(layout: LayoutSpec, config: ArrayHydrodynamicsConfig) -> list[dict[str, object]]:
    geometries = config.layout.module_geometries(config.module.length_m)
    mappings = module_structural_node_mappings(config)
    rows = []
    for index, (geometry, mapping) in enumerate(zip(geometries, mappings), start=1):
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_id": index,
                "module_length_m": geometry.length_m,
                "x_start_m": geometry.x_start_m,
                "x_end_m": geometry.x_end_m,
                "center_x_m": geometry.x_m,
                "width_m": WIDTH_M,
                "height_m": HEIGHT_M,
                "selected_node_id": int(mapping["fem_node_one_based"]),
                "selected_node_x_m": float(mapping["x_m"]),
                "selected_node_y_m": float(mapping["y_m"]),
                "abs_error_m": abs(float(mapping["x_m"]) - geometry.x_m),
            }
        )
    return rows


def geometry_rows(layout: LayoutSpec, config: ArrayHydrodynamicsConfig) -> list[dict[str, object]]:
    if layout.is_uniform:
        return uniform_geometry_rows(layout)
    return nonuniform_geometry_rows(layout, config)


def write_geometry_csv(layout: LayoutSpec, rows: list[dict[str, object]]) -> Path:
    path = GEOMETRY_DIR / f"{layout.layout_id}_module_geometry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def master_nodes_from_geometry(rows: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(int(row["selected_node_id"]) for row in rows)


def build_case(
    *,
    layout: LayoutSpec,
    hydro_path: Path,
    master_nodes: tuple[int, ...],
    wavelength_m: int,
    frequency_index: int,
) -> RodmFrequencyCase:
    return RodmFrequencyCase(
        case_id=f"{layout.layout_id}_{wavelength_m}m_serep_ridge",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=master_nodes[0], node_interval=1, count=len(master_nodes)),
        master_nodes_one_based=master_nodes,
        hydrodynamic_dataset=hydro_path,
        structural_matrices=base.structural_paths(),
        hydrodynamic_nodes=len(master_nodes),
        hydrodynamic_dof_to_remove_zero_based=5,
        structural_reduction_method="serep_ridge",
        preserve_master_order=True,
        serep_ridge_relative_lambda=SEREP_RIDGE_RELATIVE_LAMBDA,
        frequency_index=frequency_index,
        reverse_hydrodynamic_node_order=False,
    )


def response_path(layout: LayoutSpec, wavelength_m: int) -> Path:
    return RESPONSE_DIR / layout.layout_id / f"{layout.layout_id}_wavelength_{wavelength_m}m_response.npy"


def solve_layout(
    layout: LayoutSpec,
    config: ArrayHydrodynamicsConfig,
    geometry: list[dict[str, object]],
    wavelengths_m: tuple[int, ...],
    *,
    force_response: bool,
) -> dict[int, Path]:
    master_nodes = master_nodes_from_geometry(geometry)
    paths: dict[int, Path] = {}
    for frequency_index, wavelength_m in enumerate(wavelengths_m):
        path = response_path(layout, wavelength_m)
        paths[wavelength_m] = path
        if path.exists() and not force_response:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        case = build_case(
            layout=layout,
            hydro_path=config.output_path,
            master_nodes=master_nodes,
            wavelength_m=wavelength_m,
            frequency_index=frequency_index,
        )
        response = solve_rodm_frequency_case(case).global_displacement
        np.save(path, response)
    return paths


def roughness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.max(np.abs(np.diff(values, n=2))))


def load_heave(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return extract_centerline_heave(np.load(path))


def write_metrics(
    layouts: tuple[LayoutSpec, ...],
    response_paths: dict[tuple[str, int], Path],
    wavelengths_m: tuple[int, ...],
) -> tuple[Path, Path]:
    by_wavelength_path = TABLE_DIR / "wavelength_sweep_by_wavelength.csv"
    layout_summary_path = TABLE_DIR / "wavelength_sweep_layout_summary.csv"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for wavelength_m in wavelengths_m:
        x_ref, heave_ref = load_heave(response_paths[("U30_reference", wavelength_m)])
        _, heave_u10 = load_heave(response_paths[("uniform_U10", wavelength_m)])
        u10_rmse = float(np.sqrt(np.mean((heave_u10 - heave_ref) ** 2)))
        for layout in layouts:
            x, heave = load_heave(response_paths[(layout.layout_id, wavelength_m)])
            if x.shape != x_ref.shape or not np.allclose(x, x_ref):
                raise ValueError(f"centerline mismatch for {layout.layout_id} at {wavelength_m} m")
            delta = heave - heave_ref
            rmse = float(np.sqrt(np.mean(delta**2)))
            max_abs = float(np.max(np.abs(delta)))
            improvement = (u10_rmse - rmse) / u10_rmse * 100.0 if u10_rmse else 0.0
            rows.append(
                {
                    "layout_id": layout.layout_id,
                    "display_name": layout.display_name,
                    "category": layout.category,
                    "wavelength_m": wavelength_m,
                    "rmse_vs_U30": rmse,
                    "max_abs_vs_U30": max_abs,
                    "roughness": roughness(heave),
                    "improvement_vs_U10_percent": improvement,
                    "response_path": str(response_paths[(layout.layout_id, wavelength_m)]),
                }
            )

    with by_wavelength_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for layout in layouts:
        layout_rows = [row for row in rows if row["layout_id"] == layout.layout_id]
        geometry = geometry_rows(layout, hydro_config(layout, wavelengths_m, n_jobs=1))
        node_ids = [int(row["selected_node_id"]) for row in geometry]
        summary_rows.append(
            {
                "layout_id": layout.layout_id,
                "display_name": layout.display_name,
                "category": layout.category,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "mean_rmse_vs_U30": float(np.mean([row["rmse_vs_U30"] for row in layout_rows])),
                "max_rmse_vs_U30": float(np.max([row["rmse_vs_U30"] for row in layout_rows])),
                "mean_improvement_vs_U10_percent": float(
                    np.mean([row["improvement_vs_U10_percent"] for row in layout_rows])
                ),
                "better_than_U10_count": sum(
                    1
                    for row in layout_rows
                    if layout.layout_id not in {"U30_reference", "uniform_U10"}
                    and float(row["improvement_vs_U10_percent"]) > 0.0
                ),
                "selected_node_ids": " ".join(str(value) for value in node_ids),
                "max_control_point_abs_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
            }
        )
    with layout_summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return by_wavelength_path, layout_summary_path


def write_best_by_wavelength(metrics_csv: Path, wavelengths_m: tuple[int, ...]) -> Path:
    rows = read_csv(metrics_csv)
    path = TABLE_DIR / "wavelength_sweep_best_by_wavelength.csv"
    output_rows = []
    for wavelength_m in wavelengths_m:
        candidates = [
            row
            for row in rows
            if int(row["wavelength_m"]) == wavelength_m and row["layout_id"] != "U30_reference"
        ]
        best = min(candidates, key=lambda row: float(row["rmse_vs_U30"]))
        uniform = next(row for row in candidates if row["layout_id"] == "uniform_U10")
        output_rows.append(
            {
                "wavelength_m": wavelength_m,
                "best_layout_id": best["layout_id"],
                "best_display_name": best["display_name"],
                "best_rmse_vs_U30": float(best["rmse_vs_U30"]),
                "uniform_U10_rmse_vs_U30": float(uniform["rmse_vs_U30"]),
                "best_improvement_vs_U10_percent": float(best["improvement_vs_U10_percent"]),
                "best_max_abs_vs_U30": float(best["max_abs_vs_U30"]),
                "uniform_U10_max_abs_vs_U30": float(uniform["max_abs_vs_U30"]),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric_lookup(rows: list[dict[str, str]], layout_id: str, wavelength_m: int, key: str) -> float:
    for row in rows:
        if row["layout_id"] == layout_id and int(row["wavelength_m"]) == wavelength_m:
            return float(row[key])
    raise KeyError((layout_id, wavelength_m, key))


def plot_applicability_heatmap(metrics_csv: Path, layouts: tuple[LayoutSpec, ...], wavelengths_m: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "wavelength_sweep_applicability_heatmap.png"
    rows = read_csv(metrics_csv)
    plot_layouts = [layout for layout in layouts if layout.layout_id not in {"U30_reference", "uniform_U10"}]
    values = np.asarray(
        [
            [metric_lookup(rows, layout.layout_id, wavelength_m, "improvement_vs_U10_percent") for wavelength_m in wavelengths_m]
            for layout in plot_layouts
        ]
    )
    vmax = max(1.0, float(np.max(np.abs(values))))

    fig, axis = plt.subplots(figsize=(12.0, 4.8))
    image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(wavelengths_m)))
    axis.set_xticklabels([str(value) for value in wavelengths_m])
    axis.set_yticks(np.arange(len(plot_layouts)))
    axis.set_yticklabels([layout.display_name for layout in plot_layouts])
    axis.set_xlabel("wavelength (m)")
    axis.set_title("Applicability map: RMSE improvement vs uniform U10 (%)")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means NU10 is closer to U30")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_rmse_curves(metrics_csv: Path, layouts: tuple[LayoutSpec, ...], wavelengths_m: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "wavelength_sweep_rmse_curves.png"
    rows = read_csv(metrics_csv)
    fig, axis = plt.subplots(figsize=(11.2, 5.8))
    for layout in layouts:
        if layout.layout_id == "U30_reference":
            continue
        values = [metric_lookup(rows, layout.layout_id, wavelength_m, "rmse_vs_U30") for wavelength_m in wavelengths_m]
        axis.plot(
            wavelengths_m,
            values,
            marker="o",
            linewidth=1.6,
            color=COLORS.get(layout.layout_id, "#666666"),
            label=layout.display_name,
        )
    axis.set_xlabel("wavelength (m)")
    axis.set_ylabel("heave RMSE vs U30")
    axis.set_title("Extended wavelength sweep: heave RMSE")
    axis.set_xticks(wavelengths_m)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_heave_grid(
    response_paths: dict[tuple[str, int], Path],
    layouts: tuple[LayoutSpec, ...],
    wavelengths_m: tuple[int, ...],
) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "wavelength_sweep_heave_grid.png"
    column_count = 3
    row_count = int(np.ceil(len(wavelengths_m) / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(15.0, 3.6 * row_count), sharex=True)
    axes_flat = np.ravel(axes)
    for axis, wavelength_m in zip(axes_flat, wavelengths_m):
        x_ref, heave_ref = load_heave(response_paths[("U30_reference", wavelength_m)])
        axis.plot(x_ref, heave_ref, color="#111111", linewidth=2.0, label="U30 reference")
        for layout in layouts:
            if layout.layout_id == "U30_reference":
                continue
            _, heave = load_heave(response_paths[(layout.layout_id, wavelength_m)])
            axis.plot(
                x_ref,
                heave,
                linewidth=1.15,
                linestyle="-" if layout.layout_id == "uniform_U10" else "--",
                color=COLORS.get(layout.layout_id, "#666666"),
                label=layout.display_name,
            )
        axis.set_title(f"{wavelength_m} m")
        axis.set_ylabel("Heave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    for axis in axes_flat[len(wavelengths_m) :]:
        axis.axis("off")
    axes_flat[0].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes_flat[-column_count:]:
        axis.set_xlabel("x/L")
    fig.suptitle("Extended wavelength sweep: heave response curves", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    *,
    wavelengths_m: tuple[int, ...],
    metrics_csv: Path,
    layout_summary_csv: Path,
    best_by_wavelength_csv: Path,
    geometry_paths: dict[str, Path],
    figures: dict[str, Path],
) -> Path:
    metrics = read_csv(metrics_csv)
    summary = read_csv(layout_summary_csv)
    best_by_wavelength = read_csv(best_by_wavelength_csv)
    heatmap_rows = []
    for layout in selected_layouts(include_searched=True):
        if layout.layout_id in {"U30_reference", "uniform_U10"}:
            continue
        if not any(row["layout_id"] == layout.layout_id for row in metrics):
            continue
        improvements = [
            metric_lookup(metrics, layout.layout_id, wavelength_m, "improvement_vs_U10_percent")
            for wavelength_m in wavelengths_m
        ]
        heatmap_rows.append(
            (
                layout.display_name,
                f"{np.mean(improvements):.2f}%",
                sum(1 for value in improvements if value > 0.0),
                f"{max(improvements):.2f}%",
                wavelengths_m[int(np.argmax(improvements))],
            )
        )

    layout_rows = []
    for row in summary:
        if row["layout_id"] == "U30_reference":
            continue
        layout_rows.append(
            (
                row["display_name"],
                f"`[{', '.join(row['module_lengths_m'].split())}]`",
                f"{float(row['mean_rmse_vs_U30']):.6g}",
                f"{float(row['mean_improvement_vs_U10_percent']):.2f}%",
                row["better_than_U10_count"],
                row["max_control_point_abs_error_m"],
            )
        )
    best_rows = [
        (
            row["wavelength_m"],
            row["best_display_name"],
            f"{float(row['best_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in best_by_wavelength
    ]

    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# SEREP-ridge 非均匀模块波长适用性扫描",
        "",
        "## 1. 研究目的",
        "",
        "本实验冻结代表性布局，不再搜索新布局，然后把波长采样从五个典型点扩展为更密集的扫描。目标是回答：非均匀 NU10 在哪些波长范围内比均匀 U10 更接近 U30 参考解。",
        "",
        f"扫描波长为：`{', '.join(str(value) for value in wavelengths_m)} m`。参考解仍为 `U30 SEREP-ridge`，比较指标为 centerline heave RAO 相对 U30 的 RMSE、最大绝对误差和 roughness。",
        "",
        "## 2. 适用性地图",
        "",
        f"![适用性热图](figures/{figures['heatmap'].name})",
        "",
        table(("layout", "mean improvement", "better count", "best improvement", "best wavelength (m)"), heatmap_rows),
        "",
        "热图中的正值表示对应 NU10 比均匀 U10 更接近 U30。该图是后续论文讨论的关键图：它可以把“非均匀有时更好”推进为“非均匀在什么目标波长下更有价值”。",
        "",
        "逐波长最佳方案如下：",
        "",
        table(("wavelength (m)", "best case", "best RMSE", "U10 RMSE", "improvement"), best_rows),
        "",
        "## 3. RMSE 曲线",
        "",
        f"![RMSE 曲线](figures/{figures['rmse'].name})",
        "",
        table(
            ("case", "module lengths (m)", "mean RMSE", "mean improvement", "better count", "max node error (m)"),
            layout_rows,
        ),
        "",
        "## 4. Heave 响应曲线",
        "",
        f"![Heave 曲线](figures/{figures['heave'].name})",
        "",
        "该图用于检查加密波长下是否出现新的毛刺或相位异常。如果曲线保持平滑，说明非均匀模块、主控制点顺序和 SEREP-ridge 降维在更密集频率采样下仍然一致。",
        "",
        "## 5. 输出文件",
        "",
        f"- 按波长指标：`{metrics_csv}`",
        f"- 逐波长最佳方案：`{best_by_wavelength_csv}`",
        f"- 布局汇总：`{layout_summary_csv}`",
        "- 几何/控制点表：",
    ]
    for layout_id, path in geometry_paths.items():
        lines.append(f"  - `{layout_id}`: `{path}`")
    lines.extend(
        [
            f"- 图片目录：`{FIGURE_DIR}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return REPORT_PATH


def write_dry_run_plan(
    layouts: tuple[LayoutSpec, ...],
    configs: dict[str, ArrayHydrodynamicsConfig],
    geometry_paths: dict[str, Path],
    wavelengths_m: tuple[int, ...],
) -> Path:
    plan = {
        "wavelengths_m": list(wavelengths_m),
        "layouts": [
            {
                "layout_id": layout.layout_id,
                "display_name": layout.display_name,
                "category": layout.category,
                "module_lengths_m": list(layout.module_lengths_m),
                "hydro_path": str(configs[layout.layout_id].output_path),
                "hydro_exists": configs[layout.layout_id].output_path.exists(),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
                "response_paths": [str(response_path(layout, wavelength_m)) for wavelength_m in wavelengths_m],
            }
            for layout in layouts
        ],
    }
    path = OUTPUT_ROOT / "wavelength_sweep_dry_run_plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_dry_run_report(
    layouts: tuple[LayoutSpec, ...],
    configs: dict[str, ArrayHydrodynamicsConfig],
    geometry_paths: dict[str, Path],
    wavelengths_m: tuple[int, ...],
    plan_path: Path,
) -> Path:
    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    rows = []
    control_rows = []
    for layout in layouts:
        geometry = geometry_rows(layout, configs[layout.layout_id])
        node_ids = [int(row["selected_node_id"]) for row in geometry]
        rows.append(
            (
                layout.display_name,
                layout.category,
                f"`[{', '.join(str(int(value)) for value in layout.module_lengths_m)}]`",
                str(configs[layout.layout_id].output_path.exists()),
                f"`{configs[layout.layout_id].output_path}`",
            )
        )
        control_rows.append(
            (
                layout.layout_id,
                len(geometry),
                f"{sum(float(row['module_length_m']) for row in geometry):.1f}",
                ", ".join(str(value) for value in node_ids),
                f"{max(float(row['abs_error_m']) for row in geometry):.1f}",
                "no" if len(set(node_ids)) == len(node_ids) else "yes",
            )
        )

    run_command = (
        "D:\\anaconda\\envs\\capytaine\\python.exe "
        "scripts\\run_serep_nonuniform_wavelength_sweep.py --include-searched-layout"
    )
    lines = [
        "# SEREP-ridge 非均匀模块波长适用性扫描实验方案",
        "",
        "## 1. 实验目的",
        "",
        "本实验用于把当前五个典型波长的验证扩展为更密集的波长扫描。布局不再重新搜索，而是固定代表性 NU10 布局，比较其 heave 响应相对 `U30 SEREP-ridge` 参考解的误差。",
        "",
        f"波长集合：`{', '.join(str(value) for value in wavelengths_m)} m`。",
        "",
        "## 2. 固定布局",
        "",
        table(("case", "type", "module lengths (m)", "hydro exists", "hydro path"), rows),
        "",
        "## 3. 控制点核查",
        "",
        table(("layout", "module count", "total length (m)", "FEM node ids", "max x error (m)", "duplicate nodes"), control_rows),
        "",
        "所有 dry-run 几何表均已写出。正式运行时会先生成对应 `.nc` 水动力文件，然后逐波长运行 RODM，并生成适用性热图、RMSE 曲线和 heave 响应图。",
        "",
        "## 4. 正式运行命令",
        "",
        "```powershell",
        run_command,
        "```",
        "",
        "若要只运行规则型布局而不包含搜索得到的 `NU10_mean_best`，去掉 `--include-searched-layout` 即可。",
        "",
        "## 5. Dry-run 输出",
        "",
        f"- JSON 执行计划：`{plan_path}`",
        "- 几何/控制点表：",
    ]
    for layout_id, path in geometry_paths.items():
        lines.append(f"  - `{layout_id}`: `{path}`")
    lines.append("")
    path = OUTPUT_ROOT / "wavelength_sweep_experiment_plan.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def write_manifest(outputs: dict[str, Path]) -> Path:
    path = OUTPUT_ROOT / "wavelength_sweep_manifest.json"
    path.write_text(
        json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_workflow(args: argparse.Namespace) -> None:
    wavelengths_m = parse_wavelengths(args.wavelengths)
    layouts = selected_layouts(include_searched=args.include_searched_layout)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HYDRO_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    configs = {layout.layout_id: hydro_config(layout, wavelengths_m, n_jobs=args.n_jobs) for layout in layouts}
    geometry_by_layout = {layout.layout_id: geometry_rows(layout, configs[layout.layout_id]) for layout in layouts}
    geometry_paths = {layout.layout_id: write_geometry_csv(layout, geometry_by_layout[layout.layout_id]) for layout in layouts}

    if args.dry_run:
        plan_path = write_dry_run_plan(layouts, configs, geometry_paths, wavelengths_m)
        report_path = write_dry_run_report(layouts, configs, geometry_paths, wavelengths_m, plan_path)
        print(f"dry_run_plan={plan_path}")
        print(f"dry_run_report={report_path}")
        print(f"geometry_dir={GEOMETRY_DIR}")
        return

    response_paths: dict[tuple[str, int], Path] = {}
    for layout in layouts:
        config = configs[layout.layout_id]
        ensure_hydrodynamics(config, force=args.force_hydro)
        solved = solve_layout(
            layout,
            config,
            geometry_by_layout[layout.layout_id],
            wavelengths_m,
            force_response=args.force_response,
        )
        for wavelength_m, path in solved.items():
            response_paths[(layout.layout_id, wavelength_m)] = path

    metrics_csv, layout_summary_csv = write_metrics(layouts, response_paths, wavelengths_m)
    best_by_wavelength_csv = write_best_by_wavelength(metrics_csv, wavelengths_m)
    figures = {
        "heatmap": plot_applicability_heatmap(metrics_csv, layouts, wavelengths_m),
        "rmse": plot_rmse_curves(metrics_csv, layouts, wavelengths_m),
        "heave": plot_heave_grid(response_paths, layouts, wavelengths_m),
    }
    report = write_report(
        wavelengths_m=wavelengths_m,
        metrics_csv=metrics_csv,
        layout_summary_csv=layout_summary_csv,
        best_by_wavelength_csv=best_by_wavelength_csv,
        geometry_paths=geometry_paths,
        figures=figures,
    )
    manifest = write_manifest(
        {
            "report": report,
            "metrics": metrics_csv,
            "best_by_wavelength": best_by_wavelength_csv,
            "layout_summary": layout_summary_csv,
            **{f"figure_{key}": value for key, value in figures.items()},
        }
    )
    print(f"report={report}")
    print(f"metrics={metrics_csv}")
    print(f"figures={FIGURE_DIR}")
    print(f"manifest={manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wavelengths",
        default=None,
        help="Comma/space separated wavelengths in m. Default: 60,90,...,300.",
    )
    parser.add_argument("--include-searched-layout", action="store_true", help="Include NU10 mean-best searched layout.")
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate hydrodynamic NC files.")
    parser.add_argument("--force-response", action="store_true", help="Recompute RODM response npy files.")
    parser.add_argument("--n-jobs", type=int, default=CAPYTAINE_N_JOBS, help="Capytaine worker count.")
    parser.add_argument("--dry-run", action="store_true", help="Only write geometry and an execution plan.")
    return parser.parse_args()


def main() -> int:
    run_workflow(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
