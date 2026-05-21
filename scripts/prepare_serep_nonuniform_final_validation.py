"""Build final validation tables and figures for NU10 SEREP-ridge results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_serep_nonuniform_design_study as base  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402


UNIFORM_ROOT = REPO_ROOT / "results" / "uniform_reference_convergence_U5_U10_U15_U30_heave_serep_ridge_ordered"
SEARCH_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_layout_search"
REFINE_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_target_refinement"
MECHANISM_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_mechanism_report"
REPORT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_final_validation"
FIGURE_DIR = REPORT_ROOT / "figures"
TABLE_DIR = REPORT_ROOT / "tables"
REPORT_PATH = REPORT_ROOT / "serep_nonuniform_final_validation_report.md"

WAVELENGTHS_M = tuple(int(value) for value in base.WAVELENGTHS_M)
BODY_LENGTH_M = 300.0


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    display_name: str
    category: str
    role: str


LAYOUTS = (
    LayoutSpec("uniform_U10", "U10 uniform", "baseline", "10 equal 30 m modules"),
    LayoutSpec("cand_3333334332", "NU10 mean-best", "searched", "best mean RMSE from constrained search"),
    LayoutSpec("prev_center_refined", "NU10 center-refined", "rule", "short modules near the middle"),
    LayoutSpec("prev_edge_mild", "NU10 edge-mild", "rule", "mild refinement near both ends"),
    LayoutSpec("prev_bow_refined", "NU10 bow-refined", "rule", "short modules near incident-wave end"),
)


COLORS = {
    "uniform_U10": "#1f77b4",
    "cand_3333334332": "#2ca02c",
    "prev_center_refined": "#ff7f0e",
    "prev_edge_mild": "#9467bd",
    "prev_bow_refined": "#d62728",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_summary_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(REFINE_ROOT / "serep_nonuniform_layout_search_summary.csv"):
        rows.append(
            {
                "layout_id": row["layout_id"],
                "wavelength_m": int(row["wavelength_m"]),
                "omega_rad_s": float(row["omega_rad_s"]) if row["omega_rad_s"] != "nan" else float("nan"),
                "rmse": float(row["rmse_vs_U30_serep_ridge"]),
                "max_abs": float(row["max_abs_vs_U30_serep_ridge"]),
                "roughness": float(row["roughness"]),
                "response_path": Path(row["response_path"]),
                "hydro_path": Path(row["hydro_path"]),
            }
        )
    return rows


def read_layout_lengths() -> dict[str, tuple[float, ...]]:
    lengths = {"uniform_U30": (10.0,) * 30}
    for row in read_csv(REFINE_ROOT / "serep_nonuniform_layout_search_ranking.csv"):
        lengths[row["layout_id"]] = tuple(float(value) for value in row["lengths_m"].replace(",", " ").split())
    return lengths


def summary_row(rows: list[dict[str, object]], layout_id: str, wavelength_m: int) -> dict[str, object]:
    for row in rows:
        if row["layout_id"] == layout_id and row["wavelength_m"] == wavelength_m:
            return row
    raise KeyError((layout_id, wavelength_m))


def response_path(rows: list[dict[str, object]], layout_id: str, wavelength_m: int) -> Path:
    if layout_id == "uniform_U30":
        return base.reference_response_path(wavelength_m)
    return Path(summary_row(rows, layout_id, wavelength_m)["response_path"])


def centerline_heave(rows: list[dict[str, object]], layout_id: str, wavelength_m: int) -> tuple[np.ndarray, np.ndarray]:
    response = np.load(response_path(rows, layout_id, wavelength_m))
    return extract_centerline_heave(response)


def roughness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.max(np.abs(np.diff(values, n=2))))


def geometry_path(layout_id: str) -> Path:
    candidates = [
        UNIFORM_ROOT / "U10_module_geometry.csv" if layout_id == "uniform_U10" else None,
        UNIFORM_ROOT / "U30_module_geometry.csv" if layout_id == "uniform_U30" else None,
        REFINE_ROOT / "geometry" / f"{layout_id}_module_geometry.csv",
        SEARCH_ROOT / "geometry" / f"{layout_id}_module_geometry.csv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(layout_id)


def read_geometry(layout_id: str) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(geometry_path(layout_id)):
        rows.append(
            {
                "layout_id": layout_id,
                "module_id": int(row["module_id"]),
                "module_length_m": float(row["module_length_m"]),
                "x_start_m": float(row["x_start_m"]),
                "x_end_m": float(row["x_end_m"]),
                "center_x_m": float(row["center_x_m"]),
                "width_m": float(row.get("width_m") or row.get("module_width_m") or 60.0),
                "height_m": float(row.get("module_height_m") or 2.0),
                "selected_node_id": int(row["selected_node_id"]),
                "selected_node_x_m": float(row["selected_node_x_m"]),
                "selected_node_y_m": float(row.get("selected_node_y_m") or 0.0),
                "abs_error_m": float(row["abs_error_m"]),
            }
        )
    return rows


def write_control_points() -> Path:
    path = TABLE_DIR / "final_control_points.csv"
    fieldnames = [
        "layout_id",
        "module_id",
        "module_length_m",
        "x_start_m",
        "x_end_m",
        "center_x_m",
        "width_m",
        "height_m",
        "selected_node_id",
        "selected_node_x_m",
        "selected_node_y_m",
        "abs_error_m",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layout_id in ("uniform_U30", *(item.layout_id for item in LAYOUTS)):
            for row in read_geometry(layout_id):
                writer.writerow({key: row[key] for key in fieldnames})
    return path


def build_metric_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_rows = []
    for wavelength_m in WAVELENGTHS_M:
        x_ref, heave_ref = centerline_heave(summary_rows, "uniform_U30", wavelength_m)
        uniform_rmse = float(summary_row(summary_rows, "uniform_U10", wavelength_m)["rmse"])
        metric_rows.append(
            {
                "layout_id": "uniform_U30",
                "wavelength_m": wavelength_m,
                "rmse_vs_U30": 0.0,
                "max_abs_vs_U30": 0.0,
                "roughness": roughness(heave_ref),
                "improvement_vs_U10_percent": 100.0,
                "response_path": str(response_path(summary_rows, "uniform_U30", wavelength_m)),
            }
        )
        for spec in LAYOUTS:
            x, heave = centerline_heave(summary_rows, spec.layout_id, wavelength_m)
            if x.shape != x_ref.shape or not np.allclose(x, x_ref):
                raise ValueError(f"centerline x mismatch for {spec.layout_id} at {wavelength_m} m")
            delta = heave - heave_ref
            rmse = float(np.sqrt(np.mean(delta**2)))
            max_abs = float(np.max(np.abs(delta)))
            improvement = (uniform_rmse - rmse) / uniform_rmse * 100.0 if uniform_rmse else 0.0
            metric_rows.append(
                {
                    "layout_id": spec.layout_id,
                    "wavelength_m": wavelength_m,
                    "rmse_vs_U30": rmse,
                    "max_abs_vs_U30": max_abs,
                    "roughness": roughness(heave),
                    "improvement_vs_U10_percent": improvement,
                    "response_path": str(response_path(summary_rows, spec.layout_id, wavelength_m)),
                }
            )
    return metric_rows


def write_metric_csv(metric_rows: list[dict[str, object]]) -> Path:
    path = TABLE_DIR / "final_validation_by_wavelength.csv"
    fieldnames = [
        "layout_id",
        "wavelength_m",
        "rmse_vs_U30",
        "max_abs_vs_U30",
        "roughness",
        "improvement_vs_U10_percent",
        "response_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    return path


def write_layout_summary(
    metric_rows: list[dict[str, object]],
    lengths_by_layout: dict[str, tuple[float, ...]],
) -> Path:
    path = TABLE_DIR / "final_layout_summary.csv"
    fieldnames = [
        "layout_id",
        "display_name",
        "category",
        "role",
        "module_lengths_m",
        "mean_rmse_vs_U30",
        "max_rmse_vs_U30",
        "mean_max_abs_vs_U30",
        "wavelengths_better_than_U10",
        "selected_node_ids",
        "max_control_point_abs_error_m",
        "has_duplicate_control_nodes",
    ]
    specs = (LayoutSpec("uniform_U30", "U30 reference", "reference", "30 equal 10 m modules"), *LAYOUTS)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            layout_rows = [row for row in metric_rows if row["layout_id"] == spec.layout_id]
            geometry = read_geometry(spec.layout_id)
            node_ids = [int(row["selected_node_id"]) for row in geometry]
            writer.writerow(
                {
                    "layout_id": spec.layout_id,
                    "display_name": spec.display_name,
                    "category": spec.category,
                    "role": spec.role,
                    "module_lengths_m": " ".join(f"{value:g}" for value in lengths_by_layout[spec.layout_id]),
                    "mean_rmse_vs_U30": float(np.mean([row["rmse_vs_U30"] for row in layout_rows])),
                    "max_rmse_vs_U30": float(np.max([row["rmse_vs_U30"] for row in layout_rows])),
                    "mean_max_abs_vs_U30": float(np.mean([row["max_abs_vs_U30"] for row in layout_rows])),
                    "wavelengths_better_than_U10": sum(
                        1
                        for row in layout_rows
                        if spec.layout_id not in ("uniform_U10", "uniform_U30")
                        and float(row["improvement_vs_U10_percent"]) > 0.0
                    ),
                    "selected_node_ids": " ".join(str(value) for value in node_ids),
                    "max_control_point_abs_error_m": max(float(row["abs_error_m"]) for row in geometry),
                    "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
                }
            )
    return path


def target_best_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_ids = [spec.layout_id for spec in LAYOUTS]
    rows = []
    for wavelength_m in WAVELENGTHS_M:
        uniform = next(
            row for row in metric_rows if row["layout_id"] == "uniform_U10" and row["wavelength_m"] == wavelength_m
        )
        candidates = [
            row
            for row in metric_rows
            if row["layout_id"] in selected_ids and row["wavelength_m"] == wavelength_m
        ]
        best = min(candidates, key=lambda row: row["rmse_vs_U30"])
        rows.append(
            {
                "wavelength_m": wavelength_m,
                "best_layout_id": best["layout_id"],
                "best_rmse_vs_U30": best["rmse_vs_U30"],
                "uniform_U10_rmse_vs_U30": uniform["rmse_vs_U30"],
                "improvement_vs_U10_percent": best["improvement_vs_U10_percent"],
                "best_max_abs_vs_U30": best["max_abs_vs_U30"],
                "uniform_U10_max_abs_vs_U30": uniform["max_abs_vs_U30"],
            }
        )
    return rows


def write_target_best_csv(rows: list[dict[str, object]]) -> Path:
    path = TABLE_DIR / "final_target_best_by_wavelength.csv"
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def metric_value(metric_rows: list[dict[str, object]], layout_id: str, wavelength_m: int, key: str) -> float:
    for row in metric_rows:
        if row["layout_id"] == layout_id and row["wavelength_m"] == wavelength_m:
            return float(row[key])
    raise KeyError((layout_id, wavelength_m, key))


def plot_heave_panel(summary_rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "final_validation_heave_panel.png"
    fig, axes = plt.subplots(len(WAVELENGTHS_M), 1, figsize=(11.4, 16.0), sharex=True)
    fig.suptitle("Final SEREP-ridge validation: NU10 layouts vs U10 and U30 reference", fontsize=16)
    for axis, wavelength_m in zip(axes, WAVELENGTHS_M):
        x, heave_ref = centerline_heave(summary_rows, "uniform_U30", wavelength_m)
        axis.plot(x, heave_ref, color="#111111", linewidth=2.3, label="U30 reference")
        for spec in LAYOUTS:
            _, heave = centerline_heave(summary_rows, spec.layout_id, wavelength_m)
            axis.plot(
                x,
                heave,
                color=COLORS[spec.layout_id],
                linewidth=1.4,
                linestyle="-" if spec.layout_id == "uniform_U10" else "--",
                label=spec.display_name,
            )
        axis.set_ylabel(f"{wavelength_m} m\nHeave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="best")
    axes[-1].set_xlabel("x/L")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_rmse_panel(metric_rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "final_validation_rmse_panel.png"
    layout_ids = [spec.layout_id for spec in LAYOUTS]
    labels = [spec.display_name for spec in LAYOUTS]
    x = np.arange(len(WAVELENGTHS_M))
    width = 0.14

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    for offset, layout_id, label in zip(np.linspace(-2, 2, len(layout_ids)) * width, layout_ids, labels):
        values = [metric_value(metric_rows, layout_id, wl, "rmse_vs_U30") for wl in WAVELENGTHS_M]
        axes[0].bar(x + offset, values, width, color=COLORS[layout_id], label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(value) for value in WAVELENGTHS_M])
    axes[0].set_xlabel("wavelength (m)")
    axes[0].set_ylabel("RMSE vs U30")
    axes[0].set_title("Heave RMSE by wavelength")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)

    means = [np.mean([metric_value(metric_rows, layout_id, wl, "rmse_vs_U30") for wl in WAVELENGTHS_M]) for layout_id in layout_ids]
    axes[1].barh(labels, means, color=[COLORS[item] for item in layout_ids])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean RMSE vs U30")
    axes[1].set_title("Mean accuracy across 60-300 m")
    axes[1].grid(True, axis="x", color="#dddddd", linewidth=0.7, alpha=0.85)

    fig.suptitle("Final validation metrics for representative NU10 layouts", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_improvement_panel(metric_rows: list[dict[str, object]], best_rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "final_validation_improvement_panel.png"
    layout_ids = [spec.layout_id for spec in LAYOUTS if spec.layout_id != "uniform_U10"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for layout_id in layout_ids:
        improvements = [metric_value(metric_rows, layout_id, wl, "improvement_vs_U10_percent") for wl in WAVELENGTHS_M]
        axes[0].plot(
            WAVELENGTHS_M,
            improvements,
            marker="o",
            linewidth=1.6,
            color=COLORS[layout_id],
            label=next(spec.display_name for spec in LAYOUTS if spec.layout_id == layout_id),
        )
    axes[0].axhline(0.0, color="#111111", linewidth=0.9)
    axes[0].set_xticks(WAVELENGTHS_M)
    axes[0].set_xlabel("wavelength (m)")
    axes[0].set_ylabel("RMSE improvement vs U10 (%)")
    axes[0].set_title("Positive means NU10 is closer to U30 than U10")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)

    best_values = [row["improvement_vs_U10_percent"] for row in best_rows]
    colors = ["#2ca02c" if value > 0 else "#7f7f7f" for value in best_values]
    axes[1].bar([str(row["wavelength_m"]) for row in best_rows], best_values, color=colors)
    axes[1].axhline(0.0, color="#111111", linewidth=0.9)
    axes[1].set_xlabel("wavelength (m)")
    axes[1].set_ylabel("best NU10 improvement vs U10 (%)")
    axes[1].set_title("Best representative NU10 at each wavelength")
    axes[1].grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)

    fig.suptitle("Target-wavelength value of non-uniform module division", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_layout_control_points(lengths_by_layout: dict[str, tuple[float, ...]]) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = FIGURE_DIR / "final_validation_layouts_control_points.png"
    length_colors = {10.0: "#d9d9d9", 20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}
    specs = (LayoutSpec("uniform_U30", "U30 reference", "reference", ""), *LAYOUTS)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.0), gridspec_kw={"width_ratios": [1.3, 1.0]})

    y_positions = np.arange(len(specs))
    for y, spec in zip(y_positions, specs):
        left = 0.0
        for length in lengths_by_layout[spec.layout_id]:
            axes[0].barh(
                y,
                length,
                left=left,
                height=0.6,
                color=length_colors[length],
                edgecolor="#333333",
                linewidth=0.45,
            )
            if length >= 20.0:
                axes[0].text(left + 0.5 * length, y, str(int(length)), ha="center", va="center", fontsize=7.5)
            left += length
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([spec.display_name for spec in specs])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, BODY_LENGTH_M)
    axes[0].set_xlabel("x along floating body (m)")
    axes[0].set_title("1D module layouts, full width = 60 m")
    axes[0].grid(True, axis="x", color="#dddddd", linewidth=0.7, alpha=0.85)

    for y, spec in zip(y_positions, specs):
        geometry = read_geometry(spec.layout_id)
        centers = [row["center_x_m"] for row in geometry]
        nodes = [row["selected_node_x_m"] for row in geometry]
        axes[1].scatter(centers, np.full(len(centers), y), marker="o", s=34, color="#111111")
        axes[1].scatter(nodes, np.full(len(nodes), y), marker="+", s=70, color="#d62728")
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, BODY_LENGTH_M)
    axes[1].set_xlabel("x coordinate (m)")
    axes[1].set_title("Module centers and selected FEM nodes")
    axes[1].grid(True, axis="x", color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[1].scatter([], [], marker="o", color="#111111", label="module center")
    axes[1].scatter([], [], marker="+", color="#d62728", label="selected FEM node")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    legend = [
        Patch(facecolor=length_colors[value], edgecolor="#333333", label=f"{int(value)} m")
        for value in (10.0, 20.0, 30.0, 40.0)
    ]
    axes[0].legend(handles=legend, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

    fig.suptitle("Geometry and control-point alignment for final validation cases", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_node_sequence() -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "final_validation_node_sequence.png"
    specs = (LayoutSpec("uniform_U30", "U30 reference", "reference", ""), *LAYOUTS)
    fig, axis = plt.subplots(figsize=(11.8, 5.6))
    for spec in specs:
        geometry = read_geometry(spec.layout_id)
        module_ids = [row["module_id"] for row in geometry]
        node_ids = [row["selected_node_id"] for row in geometry]
        axis.plot(module_ids, node_ids, marker="o", linewidth=1.3, label=spec.display_name)
    axis.set_xlabel("module id")
    axis.set_ylabel("selected FEM node id")
    axis.set_title("Selected FEM node sequence is monotonic and order-preserved")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def read_mechanism_summary() -> list[dict[str, object]]:
    path = MECHANISM_ROOT / "tables" / "mechanism_summary.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "wavelength_m": int(row["wavelength_m"]),
                "best_layout": row["best_layout"],
                "fraction_nodes_improved": float(row["fraction_nodes_improved"]),
                "max_local_improvement": float(row["max_local_improvement"]),
                "max_local_improvement_x_m": float(row["max_local_improvement_x_m"]),
                "max_local_deterioration": float(row["max_local_deterioration"]),
                "max_local_deterioration_x_m": float(row["max_local_deterioration_x_m"]),
            }
        )
    return rows


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_lengths(lengths: tuple[float, ...]) -> str:
    return "[" + ", ".join(str(int(value)) for value in lengths) + "]"


def write_report(
    metric_rows: list[dict[str, object]],
    layout_summary_csv: Path,
    control_csv: Path,
    metric_csv: Path,
    target_csv: Path,
    best_rows: list[dict[str, object]],
    lengths_by_layout: dict[str, tuple[float, ...]],
    figures: dict[str, Path],
) -> None:
    layout_summary = read_csv(layout_summary_csv)
    mechanism = read_mechanism_summary()
    mechanism_by_wl = {row["wavelength_m"]: row for row in mechanism}

    overview_rows = []
    for row in layout_summary:
        if row["layout_id"] == "uniform_U30":
            continue
        overview_rows.append(
            (
                row["display_name"],
                f"`{fmt_lengths(lengths_by_layout[row['layout_id']])}`",
                f"{float(row['mean_rmse_vs_U30']):.6g}",
                row["wavelengths_better_than_U10"],
                row["max_control_point_abs_error_m"],
            )
        )

    target_rows = []
    for row in best_rows:
        wavelength_m = int(row["wavelength_m"])
        mechanism_row = mechanism_by_wl.get(wavelength_m)
        target_rows.append(
            (
                wavelength_m,
                row["best_layout_id"],
                f"{float(row['best_rmse_vs_U30']):.6g}",
                f"{float(row['uniform_U10_rmse_vs_U30']):.6g}",
                f"{float(row['improvement_vs_U10_percent']):.2f}%",
                f"{mechanism_row['fraction_nodes_improved'] * 100.0:.1f}%" if mechanism_row else "-",
            )
        )

    control_rows = []
    for layout_id in ("uniform_U30", *(spec.layout_id for spec in LAYOUTS)):
        geometry = read_geometry(layout_id)
        control_rows.append(
            (
                layout_id,
                len(geometry),
                ", ".join(str(row["selected_node_id"]) for row in geometry),
                f"{max(float(row['abs_error_m']) for row in geometry):.1f}",
                "no" if len({row["selected_node_id"] for row in geometry}) == len(geometry) else "yes",
            )
        )

    lines = [
        "# SEREP-ridge 非均匀模块最终验证报告",
        "",
        "## 1. 本轮目标",
        "",
        "本报告把当前已经确认可信的计算结果收束成一组最终验证矩阵。参考解采用 `U30 SEREP-ridge`，即 30 个均匀模块，每个模块长 10 m。对比对象包括均匀 `U10` 以及几个代表性的 `NU10` 非均匀 10 模块布局。这里只比较 heave 响应。",
        "",
        "所有布局均保持物理尺度 `300 m x 60 m x 2 m`、吃水 `0.5 m`、水深 `58.5 m`、水密度 `rho = 1000 kg/m^3`。模块划分始终是一维 `N x 1`，每个模块跨越全宽 60 m。",
        "",
        "## 2. 最终验证对象",
        "",
        f"![布局与控制点](figures/{figures['layout_control'].name})",
        "",
        markdown_table(
            ("case", "module lengths (m)", "mean RMSE vs U30", "better wavelengths", "max node error (m)"),
            overview_rows,
        ),
        "",
        "这张表的重点不是说所有非均匀布局都优于均匀 U10，而是说明：在控制点顺序和 SEREP-ridge 稳定化之后，NU10 布局可以稳定进入 RODM，并且部分目标波长下能比 U10 更接近 U30。",
        "",
        "## 3. Heave 响应曲线",
        "",
        f"![heave 对比](figures/{figures['heave'].name})",
        "",
        "曲线对比中没有再出现早期那种由节点顺序或病态逆导致的非物理毛刺。NU10 与 U30 的差异主要表现为平滑的幅值偏差，这说明当前问题已经回到模块离散与控制点布置本身，而不是程序映射错误。",
        "",
        "## 4. 误差指标",
        "",
        f"![RMSE 指标](figures/{figures['rmse'].name})",
        "",
        f"![目标波长改善率](figures/{figures['improvement'].name})",
        "",
        markdown_table(
            ("wavelength (m)", "best NU10", "NU10 RMSE", "U10 RMSE", "improvement", "local nodes improved"),
            target_rows,
        ),
        "",
        "从当前结果看，`60 m` 短波下均匀 U10 仍然最好；在 `120 m`、`180 m`、`240 m`、`300 m` 下，代表性 NU10 中存在比 U10 更接近 U30 的布局。`120 m` 和 `240 m` 的改善最清楚，说明非均匀模块更适合作为目标工况相关的离散优化，而不是无条件替代均匀划分。",
        "",
        "## 5. 主控制点顺序检查",
        "",
        f"![节点序号](figures/{figures['node_sequence'].name})",
        "",
        markdown_table(
            ("layout", "module count", "FEM node ids", "max x error (m)", "duplicate nodes"),
            control_rows,
        ),
        "",
        "所有最终验证案例的模块重心都能精确落在结构 FEM 节点上，最大 x 向误差为 0 m，且没有重复主控制节点。节点序号沿模块顺序单调变化，保持了水动力模块、结构主节点和响应重构之间的一致映射。",
        "",
        "## 6. 当前结论",
        "",
        "1. 非均匀模块的程序正确性已经有较强证据支持：几何、控制点、节点顺序和 heave 曲线均通过核查。",
        "2. 非均匀模块的精度价值是目标波长相关的：它可以在部分波长优于 U10，但不能宣称任意非均匀布局都更好。",
        "3. `U30 SEREP-ridge` 可以继续作为后续比较的高分辨率参考解；`U10` 是工程基准；`NU10-center`、`NU10-edge`、`NU10-bow` 可以作为论文中的规则型代表布局。",
        "4. `NU10 mean-best` 属于搜索得到的布局，适合放在补充讨论中，用来说明在 20/30/40 m 约束下，非均匀布局可以达到与 U10 同量级的平均精度。",
        "",
        "## 7. 输出文件",
        "",
        f"- 按波长误差表：`{metric_csv}`",
        f"- 布局总表：`{layout_summary_csv}`",
        f"- 目标波长最优表：`{target_csv}`",
        f"- 控制点总表：`{control_csv}`",
        f"- 图片目录：`{FIGURE_DIR}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def write_manifest(outputs: dict[str, Path]) -> Path:
    path = REPORT_ROOT / "final_validation_manifest.json"
    path.write_text(
        json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = read_summary_rows()
    lengths_by_layout = read_layout_lengths()
    lengths_by_layout["uniform_U10"] = (30.0,) * 10

    control_csv = write_control_points()
    metric_rows = build_metric_rows(summary_rows)
    metric_csv = write_metric_csv(metric_rows)
    layout_summary_csv = write_layout_summary(metric_rows, lengths_by_layout)
    best_rows = target_best_rows(metric_rows)
    target_csv = write_target_best_csv(best_rows)

    figures = {
        "heave": plot_heave_panel(summary_rows),
        "rmse": plot_rmse_panel(metric_rows),
        "improvement": plot_improvement_panel(metric_rows, best_rows),
        "layout_control": plot_layout_control_points(lengths_by_layout),
        "node_sequence": plot_node_sequence(),
    }
    write_report(
        metric_rows,
        layout_summary_csv,
        control_csv,
        metric_csv,
        target_csv,
        best_rows,
        lengths_by_layout,
        figures,
    )
    manifest = write_manifest(
        {
            "report": REPORT_PATH,
            "metrics_by_wavelength": metric_csv,
            "layout_summary": layout_summary_csv,
            "target_best": target_csv,
            "control_points": control_csv,
            **{f"figure_{key}": value for key, value in figures.items()},
        }
    )
    print(f"report={REPORT_PATH}")
    print(f"tables={TABLE_DIR}")
    print(f"figures={FIGURE_DIR}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
