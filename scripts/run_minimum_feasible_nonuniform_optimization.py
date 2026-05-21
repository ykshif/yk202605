"""Build the minimum-feasible non-uniform control-point evidence package.

This script consolidates the actual RODM-in-the-loop validation results that
have already been generated:

* N5..N9 low-control candidates, which define the failure boundary.
* N10..N14 non-uniform candidates, which define the feasible region.

It does not assume a surrogate result is correct.  The tables used here are
all produced after Capytaine hydrodynamic data generation and ordered
SEREP-ridge RODM response solving.  The purpose is to answer the paper-level
question:

    What is the smallest non-uniform module/control-point count that can
    reach a prescribed heave RMSE tolerance relative to the U30 reference?
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))


OUTPUT_ROOT = REPO_ROOT / "results" / "minimum_feasible_nonuniform_optimization"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "minimum_feasible_nonuniform_optimization_report.md"

LOW_CONTROL_DENSITY_CSV = (
    REPO_ROOT / "results" / "low_control_point_nonuniform_study" / "tables" / "actual_gate_by_target_count.csv"
)
LOW_CONTROL_RESPONSE_CSV = (
    REPO_ROOT
    / "results"
    / "low_control_point_response_aware_study"
    / "tables"
    / "actual_response_aware_validation.csv"
)
HIGH_CONTROL_COMBINED_CSV = (
    REPO_ROOT
    / "results"
    / "control_point_count_gate_study_combined"
    / "tables"
    / "actual_gate_by_target_count_with_N13.csv"
)
TARGETED_REFINEMENT_CSV = (
    REPO_ROOT
    / "results"
    / "targeted_nonuniform_refinement_wl300_band"
    / "tables"
    / "targeted_refinement_actual_by_layout.csv"
)

GEOMETRY_MANIFESTS = (
    REPO_ROOT / "results" / "low_control_point_nonuniform_study" / "tables" / "gate_layout_geometry_manifest.csv",
    REPO_ROOT
    / "results"
    / "low_control_point_response_aware_study"
    / "tables"
    / "response_aware_geometry_manifest.csv",
    REPO_ROOT / "results" / "control_point_count_gate_study" / "tables" / "gate_layout_geometry_manifest.csv",
    REPO_ROOT
    / "results"
    / "control_point_count_gate_study_N13_only"
    / "tables"
    / "gate_layout_geometry_manifest.csv",
    REPO_ROOT
    / "results"
    / "targeted_nonuniform_refinement_wl300_band"
    / "tables"
    / "targeted_refinement_geometry_manifest.csv",
)

TARGETS = (
    "wl_120m",
    "wl_180m",
    "wl_240m",
    "wl_300m",
    "band_equal_120_300m",
    "band_center_120_240m",
)
TARGET_ALIASES = {
    "band_120_300m": "band_equal_120_300m",
}
COUNTS = tuple(range(5, 15))
RATIO_GATES = (1.0, 0.95, 0.9)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def normalize_target_id(target_id: str) -> str:
    return TARGET_ALIASES.get(target_id, target_id)


def as_float(value: str | float | int) -> float:
    if value == "":
        return float("nan")
    return float(value)


def file_link(path: Path) -> str:
    return path.resolve().as_posix()


def row_from_source(row: dict[str, str], source_name: str) -> dict[str, object] | None:
    target_id = normalize_target_id(row.get("target_id", ""))
    if target_id not in TARGETS:
        return None
    module_count = int(row["module_count"])
    if module_count not in COUNTS:
        return None
    ratio = as_float(row["actual_rmse_ratio_vs_U10"])
    if not np.isfinite(ratio):
        return None
    improvement = as_float(row.get("actual_improvement_vs_U10_percent", (1.0 - ratio) * 100.0))
    return {
        "target_id": target_id,
        "module_count": module_count,
        "layout_id": row["layout_id"],
        "display_name": row.get("display_name", row["layout_id"]),
        "actual_rmse_ratio_vs_U10": ratio,
        "actual_improvement_vs_U10_percent": improvement,
        "actual_rmse_vs_U30": as_float(row.get("actual_rmse_vs_U30", "")),
        "uniform_U10_rmse_vs_U30": as_float(row.get("uniform_U10_rmse_vs_U30", "")),
        "target_wavelengths_m": row.get("target_wavelengths_m", ""),
        "source": source_name,
    }


def load_actual_pool() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = (
        (LOW_CONTROL_DENSITY_CSV, "low_control_density_dp_N5_N10"),
        (LOW_CONTROL_RESPONSE_CSV, "low_control_response_aware_N5_N9"),
        (HIGH_CONTROL_COMBINED_CSV, "actual_gate_N10_N14_combined"),
        (TARGETED_REFINEMENT_CSV, "targeted_refinement_wl300_band_N11_N14"),
    )
    for path, name in sources:
        for raw in read_csv(path):
            row = row_from_source(raw, name)
            if row is not None:
                rows.append(row)

    # Add the uniform U10 baseline explicitly for every target.  Some gate
    # studies replace N10 with a non-uniform N10 candidate for one target, but
    # the baseline remains the reference level.
    for target_id in TARGETS:
        rows.append(
            {
                "target_id": target_id,
                "module_count": 10,
                "layout_id": "uniform_U10",
                "display_name": "U10 uniform baseline",
                "actual_rmse_ratio_vs_U10": 1.0,
                "actual_improvement_vs_U10_percent": 0.0,
                "actual_rmse_vs_U30": float("nan"),
                "uniform_U10_rmse_vs_U30": float("nan"),
                "target_wavelengths_m": "",
                "source": "explicit_baseline",
            }
        )
    return rows


def best_pool_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the best actual layout for each target/count pair."""

    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (str(row["target_id"]), int(row["module_count"]))
        current = by_key.get(key)
        if current is None or float(row["actual_rmse_ratio_vs_U10"]) < float(
            current["actual_rmse_ratio_vs_U10"]
        ):
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key, key=lambda item: (TARGETS.index(item[0]), item[1]))]


def load_geometry_by_layout() -> dict[str, dict[str, str]]:
    geometry: dict[str, dict[str, str]] = {}
    for path in GEOMETRY_MANIFESTS:
        for row in read_csv(path):
            layout_id = row["layout_id"]
            current = geometry.get(layout_id)
            # Prefer rows with a concrete hydrodynamic path when duplicates exist.
            if current is None or (not current.get("hydro_path") and row.get("hydro_path")):
                geometry[layout_id] = row
    return geometry


def write_candidate_pool(rows: list[dict[str, object]], geometry: dict[str, dict[str, str]]) -> Path:
    output_rows: list[dict[str, object]] = []
    for row in rows:
        info = geometry.get(str(row["layout_id"]), {})
        output_rows.append(
            {
                **row,
                "module_lengths_m": info.get("module_lengths_m", ""),
                "module_centers_m": info.get("module_centers_m", ""),
                "selected_node_ids": info.get("selected_node_ids", ""),
                "max_abs_center_node_error_m": info.get("max_abs_center_node_error_m", ""),
                "hydro_path": info.get("hydro_path", ""),
                "geometry_csv": info.get("geometry_csv", ""),
            }
        )
    return write_csv(TABLE_DIR / "actual_candidate_pool_best_by_target_count.csv", output_rows)


def minimum_feasible_rows(rows: list[dict[str, object]], geometry: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    nonuniform_rows = [row for row in rows if row["layout_id"] != "uniform_U10"]
    for target_id in TARGETS:
        target_rows = [row for row in nonuniform_rows if row["target_id"] == target_id]
        for ratio_gate in RATIO_GATES:
            feasible = [row for row in target_rows if float(row["actual_rmse_ratio_vs_U10"]) <= ratio_gate]
            if feasible:
                winner = min(
                    feasible,
                    key=lambda row: (int(row["module_count"]), float(row["actual_rmse_ratio_vs_U10"])),
                )
                info = geometry.get(str(winner["layout_id"]), {})
                selected.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_nonuniform_module_count": winner["module_count"],
                        "selected_layout_id": winner["layout_id"],
                        "actual_rmse_ratio_vs_U10": winner["actual_rmse_ratio_vs_U10"],
                        "actual_improvement_vs_U10_percent": winner["actual_improvement_vs_U10_percent"],
                        "module_lengths_m": info.get("module_lengths_m", ""),
                        "module_centers_m": info.get("module_centers_m", ""),
                        "selected_node_ids": info.get("selected_node_ids", ""),
                        "hydro_path": info.get("hydro_path", ""),
                    }
                )
            else:
                selected.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_nonuniform_module_count": "",
                        "selected_layout_id": "",
                        "actual_rmse_ratio_vs_U10": "",
                        "actual_improvement_vs_U10_percent": "",
                        "module_lengths_m": "",
                        "module_centers_m": "",
                        "selected_node_ids": "",
                        "hydro_path": "",
                    }
                )
    return selected


def plot_improvement_heatmap(rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    values = np.full((len(TARGETS), len(COUNTS)), np.nan)
    labels = [["" for _ in COUNTS] for _ in TARGETS]
    for row in rows:
        target_id = str(row["target_id"])
        if target_id not in TARGETS:
            continue
        module_count = int(row["module_count"])
        if module_count not in COUNTS:
            continue
        target_index = TARGETS.index(target_id)
        count_index = COUNTS.index(module_count)
        value = float(row["actual_improvement_vs_U10_percent"])
        values[target_index, count_index] = value
        labels[target_index][count_index] = f"{value:.1f}"

    finite = values[np.isfinite(values)]
    vmax = max(10.0, float(np.nanmax(np.abs(finite)))) if finite.size else 10.0
    vmax = min(vmax, 100.0)

    fig, axis = plt.subplots(figsize=(13.2, 6.2))
    image = axis.imshow(np.clip(values, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(COUNTS)))
    axis.set_xticklabels([f"N{value}" for value in COUNTS])
    axis.set_yticks(np.arange(len(TARGETS)))
    axis.set_yticklabels(TARGETS)
    axis.set_xlabel("module/control-point count")
    axis.set_title("Actual RODM-in-the-loop improvement vs uniform U10 (%)")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if labels[row_index][col_index]:
                axis.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means closer to U30 than uniform U10")
    fig.tight_layout()
    path = FIGURE_DIR / "minimum_feasible_actual_improvement_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_ratio_curves(rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12.0, 6.0))
    for target_id in TARGETS:
        target_rows = [row for row in rows if row["target_id"] == target_id]
        x_values = [int(row["module_count"]) for row in target_rows]
        y_values = [float(row["actual_rmse_ratio_vs_U10"]) for row in target_rows]
        axis.plot(x_values, y_values, marker="o", linewidth=1.5, label=target_id)
    for gate in RATIO_GATES:
        axis.axhline(gate, color="#555555", linestyle=":", linewidth=0.9)
        axis.text(max(COUNTS) + 0.08, gate, f"{gate:.2f}", va="center", fontsize=8)
    axis.set_xlabel("module/control-point count")
    axis.set_ylabel("actual heave RMSE ratio vs uniform U10")
    axis.set_title("Minimum feasible non-uniform control-point search")
    axis.set_xticks(COUNTS)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIGURE_DIR / "minimum_feasible_actual_ratio_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_minimum_counts(minimum_rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(TARGETS))
    width = 0.24
    for offset, gate in zip((-width, 0.0, width), RATIO_GATES):
        y_values = []
        labels = []
        for target_id in TARGETS:
            row = next(
                item
                for item in minimum_rows
                if item["target_id"] == target_id and float(item["actual_ratio_gate"]) == gate
            )
            value = row["minimum_nonuniform_module_count"]
            y_values.append(np.nan if value == "" else int(value))
            labels.append("not passed" if value == "" else f"N{value}")
        bars = axis.bar(x + offset, y_values, width=width, label=f"ratio <= {gate:g}")
        for bar, label in zip(bars, labels):
            if label == "not passed":
                continue
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, label, ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(TARGETS, rotation=18, ha="right")
    axis.set_ylabel("minimum non-uniform module/control-point count")
    axis.set_title("Minimum feasible N by target and tolerance")
    axis.set_ylim(0, max(COUNTS) + 1)
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "minimum_feasible_counts_by_target.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    *,
    pool_csv: Path,
    minimum_csv: Path,
    figures: dict[str, Path],
    minimum_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> Path:
    min_table_rows = []
    for target_id in TARGETS:
        for gate in RATIO_GATES:
            row = next(
                item
                for item in minimum_rows
                if item["target_id"] == target_id and float(item["actual_ratio_gate"]) == gate
            )
            min_table_rows.append(
                (
                    target_id,
                    gate,
                    row["minimum_nonuniform_module_count"] or "not passed",
                    row["selected_layout_id"] or "-",
                    (
                        "-"
                        if row["actual_rmse_ratio_vs_U10"] == ""
                        else f"{float(row['actual_rmse_ratio_vs_U10']):.4f}"
                    ),
                    (
                        "-"
                        if row["actual_improvement_vs_U10_percent"] == ""
                        else f"{float(row['actual_improvement_vs_U10_percent']):.2f}%"
                    ),
                )
            )

    best_table_rows = []
    for target_id in TARGETS:
        rows = [row for row in best_rows if row["target_id"] == target_id and row["layout_id"] != "uniform_U10"]
        best = min(rows, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
        best_table_rows.append(
            (
                target_id,
                f"N{best['module_count']}",
                best["layout_id"],
                f"{float(best['actual_rmse_ratio_vs_U10']):.4f}",
                f"{float(best['actual_improvement_vs_U10_percent']):.2f}%",
            )
        )

    lines = [
        "# 最小可行非均匀控制点研究：真实 RODM-in-the-loop 证据包",
        "",
        "## 1. 研究定位",
        "",
        (
            "本报告把问题从“控制点是否可以无限减少”改为“在给定误差容限下，"
            "最小可行的非均匀模块/控制点数量是多少”。所有进入汇总的候选都已经经过"
            "独立 Capytaine 水动力数据和 ordered SEREP-ridge RODM 响应验证。"
        ),
        "",
        "## 2. 关键图",
        "",
        f"![Actual improvement heatmap]({file_link(figures['heatmap'])})",
        "",
        f"![Actual ratio curves]({file_link(figures['curves'])})",
        "",
        f"![Minimum feasible counts]({file_link(figures['minimum_counts'])})",
        "",
        "## 3. 每个目标下的最佳非均匀候选",
        "",
        markdown_table(
            ("target", "best N", "best layout", "RMSE ratio vs U10", "improvement"),
            best_table_rows,
        ),
        "",
        "## 4. 给定误差门限下的最小可行非均匀 N",
        "",
        markdown_table(
            ("target", "gate", "minimum N", "layout", "RMSE ratio vs U10", "improvement"),
            min_table_rows,
        ),
        "",
        "## 5. 主要结论",
        "",
        "1. N5-N9 的少控制点非均匀布局整体失败，说明该方法存在清楚的采样下限。",
        "2. N10-N14 区间开始出现可行布局，且不同波长/频带的最小可行 N 不同。",
        "3. 对单一波长，非均匀布局更容易超过 U10；对宽频带目标，则需要更高的控制点数量。",
        "4. 论文主张应表述为“误差容限约束下的最小可行控制点选择”，而不是简单说“控制点越少越好”。",
        "",
        "## 6. 输出文件",
        "",
        f"- 实际候选池：`{pool_csv}`",
        f"- 最小可行 N 表：`{minimum_csv}`",
        f"- 图目录：`{FIGURE_DIR}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return REPORT_PATH


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    actual_rows = load_actual_pool()
    geometry = load_geometry_by_layout()
    best_rows = best_pool_rows(actual_rows)
    pool_csv = write_candidate_pool(best_rows, geometry)
    minimum_rows = minimum_feasible_rows(best_rows, geometry)
    minimum_csv = write_csv(TABLE_DIR / "minimum_feasible_nonuniform_by_target_gate.csv", minimum_rows)
    figures = {
        "heatmap": plot_improvement_heatmap(best_rows),
        "curves": plot_ratio_curves(best_rows),
        "minimum_counts": plot_minimum_counts(minimum_rows),
    }
    report = write_report(
        pool_csv=pool_csv,
        minimum_csv=minimum_csv,
        figures=figures,
        minimum_rows=minimum_rows,
        best_rows=best_rows,
    )
    manifest = {
        "report": str(report),
        "pool_csv": str(pool_csv),
        "minimum_csv": str(minimum_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (OUTPUT_ROOT / "minimum_feasible_nonuniform_optimization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> int:
    manifest = run_workflow(parse_args())
    for key, value in manifest.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
