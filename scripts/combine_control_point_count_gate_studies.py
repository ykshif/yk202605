"""Combine the main control-point gate study with the N13 supplement."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import run_minimum_control_point_rodm_validation as mcpv  # noqa: E402


MAIN_ROOT = REPO_ROOT / "results" / "control_point_count_gate_study"
N13_ROOT = REPO_ROOT / "results" / "control_point_count_gate_study_N13_only"
OUTPUT_ROOT = REPO_ROOT / "results" / "control_point_count_gate_study_combined"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "control_point_count_gate_study_combined_report.md"

TARGETS = (
    "wl_120m",
    "wl_180m",
    "wl_240m",
    "wl_300m",
    "band_equal_120_300m",
    "band_center_120_240m",
)
COUNTS = (10, 11, 12, 13, 14)
RATIO_GATES = (1.0, 0.95, 0.9)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def combined_gate_rows() -> list[dict[str, object]]:
    rows = read_csv(MAIN_ROOT / "tables" / "actual_gate_by_target_count.csv")
    rows.extend(read_csv(N13_ROOT / "tables" / "actual_gate_by_target_count.csv"))
    rows.sort(key=lambda row: (TARGETS.index(row["target_id"]), int(row["module_count"])))
    return [dict(row) for row in rows]


def minimum_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for target_id in TARGETS:
        target_rows = [row for row in rows if row["target_id"] == target_id]
        for ratio_gate in RATIO_GATES:
            feasible = [
                row
                for row in target_rows
                if float(row["actual_rmse_ratio_vs_U10"]) <= ratio_gate
            ]
            if feasible:
                selected = min(
                    feasible,
                    key=lambda row: (
                        int(row["module_count"]),
                        float(row["actual_rmse_ratio_vs_U10"]),
                    ),
                )
                output.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_module_count": selected["module_count"],
                        "selected_layout_id": selected["layout_id"],
                        "selected_actual_rmse_ratio_vs_U10": selected["actual_rmse_ratio_vs_U10"],
                        "selected_actual_improvement_vs_U10_percent": selected[
                            "actual_improvement_vs_U10_percent"
                        ],
                    }
                )
            else:
                output.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_module_count": "",
                        "selected_layout_id": "",
                        "selected_actual_rmse_ratio_vs_U10": "",
                        "selected_actual_improvement_vs_U10_percent": "",
                    }
                )
    return output


def metric_lookup(rows: list[dict[str, object]], target_id: str, count: int, key: str) -> float:
    for row in rows:
        if row["target_id"] == target_id and int(row["module_count"]) == count:
            return float(row[key])
    return float("nan")


def plot_heatmap(rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    values = np.asarray(
        [
            [metric_lookup(rows, target, count, "actual_improvement_vs_U10_percent") for count in COUNTS]
            for target in TARGETS
        ],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    vmax = min(max(10.0, float(np.nanmax(np.abs(finite)))) if finite.size else 10.0, 80.0)
    fig, axis = plt.subplots(figsize=(10.8, 5.5))
    image = axis.imshow(np.clip(values, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(COUNTS)))
    axis.set_xticklabels([f"N{count}" for count in COUNTS])
    axis.set_yticks(np.arange(len(TARGETS)))
    axis.set_yticklabels(TARGETS)
    axis.set_xlabel("module/control-point count")
    axis.set_title("Combined actual RODM gate: heave RMSE improvement vs U10 (%)")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if np.isfinite(values[row_index, col_index]):
                axis.text(
                    col_index,
                    row_index,
                    f"{values[row_index, col_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means closer to U30 than U10")
    fig.tight_layout()
    path = FIGURE_DIR / "combined_actual_gate_improvement_by_target_count.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_curves(rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11.0, 5.8))
    for target in TARGETS:
        y_values = [metric_lookup(rows, target, count, "actual_rmse_ratio_vs_U10") for count in COUNTS]
        axis.plot(COUNTS, y_values, marker="o", linewidth=1.5, label=target)
    for gate in RATIO_GATES:
        axis.axhline(gate, color="#777777", linestyle=":", linewidth=0.9)
        axis.text(max(COUNTS) + 0.05, gate, f"{gate:.2f}", va="center", fontsize=8, color="#555555")
    axis.set_xlabel("module/control-point count")
    axis.set_ylabel("target RMSE ratio vs U10")
    axis.set_title("Combined actual RODM gate curves by target")
    axis.set_xticks(COUNTS)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIGURE_DIR / "combined_actual_gate_rmse_ratio_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(gate_csv: Path, minimum_csv: Path, heatmap: Path, curves: Path, minimum: list[dict[str, object]]) -> Path:
    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    min_rows = [
        (
            row["target_id"],
            row["actual_ratio_gate"],
            row["minimum_module_count"] or "not passed",
            row["selected_layout_id"] or "-",
            "-"
            if row["selected_actual_improvement_vs_U10_percent"] == ""
            else f"{float(row['selected_actual_improvement_vs_U10_percent']):.2f}%",
        )
        for row in minimum
    ]
    lines = [
        "# 控制点数量门控研究：合并 N13 后的结果",
        "",
        "本报告合并主实验 `N10/N11/N12/N14` 与补充实验 `N13`。所有非基准候选均已重新生成对应 Capytaine `.nc` 水动力数据。",
        "",
        f"![Combined heatmap]({mcpv.file_uri(heatmap)})",
        "",
        f"![Combined curves]({mcpv.file_uri(curves)})",
        "",
        "## 最小控制点数",
        "",
        table(("target", "actual ratio gate", "minimum N", "selected layout", "improvement"), min_rows),
        "",
        "## 输出",
        "",
        f"- 合并门控表：`{gate_csv}`",
        f"- 合并最小控制点表：`{minimum_csv}`",
        f"- 图目录：`{FIGURE_DIR}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return REPORT_PATH


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = combined_gate_rows()
    minimum = minimum_rows(rows)
    gate_csv = write_csv(TABLE_DIR / "actual_gate_by_target_count_with_N13.csv", rows)
    minimum_csv = write_csv(TABLE_DIR / "actual_gate_minimum_counts_with_N13.csv", minimum)
    heatmap = plot_heatmap(rows)
    curves = plot_curves(rows)
    report = write_report(gate_csv, minimum_csv, heatmap, curves, minimum)
    manifest = {
        "report": str(report),
        "gate_csv": str(gate_csv),
        "minimum_csv": str(minimum_csv),
        "heatmap": str(heatmap),
        "curves": str(curves),
    }
    (OUTPUT_ROOT / "control_point_count_gate_study_combined_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for key, value in manifest.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
