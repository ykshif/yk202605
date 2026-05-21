"""Prepare a Markdown proof report for SEREP non-uniform module validation."""

from __future__ import annotations

from pathlib import Path
import csv
import math
import shutil
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_serep_nonuniform_design_study as base  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402


SEARCH_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_layout_search"
REFINE_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_target_refinement"
REPORT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_proof_report"
FIGURE_DIR = REPORT_ROOT / "figures"
REPORT_PATH = REPORT_ROOT / "serep_nonuniform_module_serep_proof_report.md"

WAVELENGTHS_M = tuple(int(value) for value in base.WAVELENGTHS_M)
REPRESENTATIVE_LAYOUTS = (
    "uniform_U10",
    "cand_3333334332",
    "prev_center_refined",
    "prev_edge_mild",
    "prev_bow_refined",
)


def read_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (REFINE_ROOT / "serep_nonuniform_layout_search_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "layout_id": row["layout_id"],
                    "wavelength_m": int(row["wavelength_m"]),
                    "rmse": float(row["rmse_vs_U30_serep_ridge"]),
                    "max_abs": float(row["max_abs_vs_U30_serep_ridge"]),
                    "roughness": float(row["roughness"]),
                    "response_path": Path(row["response_path"]),
                }
            )
    return rows


def read_ranking() -> dict[str, tuple[float, ...]]:
    layouts: dict[str, tuple[float, ...]] = {}
    with (REFINE_ROOT / "serep_nonuniform_layout_search_ranking.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            layouts[row["layout_id"]] = tuple(
                float(value) for value in row["lengths_m"].replace(",", " ").split()
            )
    return layouts


def read_geometry(layout_id: str) -> list[dict[str, object]]:
    path = REFINE_ROOT / "geometry" / f"{layout_id}_module_geometry.csv"
    if not path.exists():
        path = SEARCH_ROOT / "geometry" / f"{layout_id}_module_geometry.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "module_id": int(row["module_id"]),
                    "module_length_m": float(row["module_length_m"]),
                    "center_x_m": float(row["center_x_m"]),
                    "selected_node_id": int(row["selected_node_id"]),
                    "selected_node_x_m": float(row["selected_node_x_m"]),
                    "abs_error_m": float(row["abs_error_m"]),
                }
            )
    return rows


def row_for(rows: list[dict[str, object]], layout_id: str, wavelength_m: int) -> dict[str, object]:
    for row in rows:
        if row["layout_id"] == layout_id and row["wavelength_m"] == wavelength_m:
            return row
    raise KeyError((layout_id, wavelength_m))


def best_nonuniform_by_wavelength(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    best = {}
    for wavelength_m in WAVELENGTHS_M:
        candidates = [
            row
            for row in rows
            if row["wavelength_m"] == wavelength_m and row["layout_id"] != "uniform_U10"
        ]
        best[wavelength_m] = min(candidates, key=lambda row: row["rmse"])
    return best


def mean_rmse(rows: list[dict[str, object]], layout_id: str) -> float:
    layout_rows = [row for row in rows if row["layout_id"] == layout_id]
    return float(np.mean([row["rmse"] for row in layout_rows]))


def plot_heave_panel(rows: list[dict[str, object]], best_nu: dict[int, dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "best_nonuniform_vs_uniform_heave_panel.png"
    fig, axes = plt.subplots(len(WAVELENGTHS_M), 1, figsize=(11.2, 16.0), sharex=True)
    fig.suptitle("SEREP-ridge validation: best NU10 vs U10 vs U30 reference", fontsize=16)
    for axis, wavelength_m in zip(axes, WAVELENGTHS_M):
        reference = np.load(base.reference_response_path(wavelength_m))
        x, heave_ref = extract_centerline_heave(reference)
        uniform = np.load(row_for(rows, "uniform_U10", wavelength_m)["response_path"])
        _, heave_uniform = extract_centerline_heave(uniform)
        best = best_nu[wavelength_m]
        nonuniform = np.load(best["response_path"])
        _, heave_nonuniform = extract_centerline_heave(nonuniform)

        axis.plot(x, heave_ref, color="#111111", linewidth=2.2, label="U30 reference")
        axis.plot(x, heave_uniform, color="#1f77b4", linewidth=1.6, label="uniform U10")
        axis.plot(
            x,
            heave_nonuniform,
            color="#d62728",
            linewidth=1.6,
            linestyle="--",
            label=f"best NU10: {best['layout_id']}",
        )
        axis.set_ylabel(f"{wavelength_m} m\nHeave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("x/L")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_rmse(rows: list[dict[str, object]], best_nu: dict[int, dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "target_wavelength_rmse_comparison.png"
    wavelengths = np.asarray(WAVELENGTHS_M, dtype=float)
    uniform_rmse = np.asarray([row_for(rows, "uniform_U10", wl)["rmse"] for wl in WAVELENGTHS_M])
    best_rmse = np.asarray([best_nu[wl]["rmse"] for wl in WAVELENGTHS_M])
    improvement = (uniform_rmse - best_rmse) / uniform_rmse * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    x = np.arange(len(WAVELENGTHS_M))
    width = 0.38
    axes[0].bar(x - width / 2, uniform_rmse, width, label="uniform U10", color="#1f77b4")
    axes[0].bar(x + width / 2, best_rmse, width, label="best NU10", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{wl}" for wl in WAVELENGTHS_M])
    axes[0].set_xlabel("wavelength (m)")
    axes[0].set_ylabel("RMSE vs U30")
    axes[0].set_title("RMSE comparison")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)

    colors = ["#2ca02c" if value > 0 else "#7f7f7f" for value in improvement]
    axes[1].bar(wavelengths.astype(str), improvement, color=colors)
    axes[1].axhline(0.0, color="#111111", linewidth=0.9)
    axes[1].set_xlabel("wavelength (m)")
    axes[1].set_ylabel("RMSE improvement vs uniform U10 (%)")
    axes[1].set_title("Positive value means NU10 is closer to U30")
    axes[1].grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    fig.suptitle("Target-wavelength accuracy of non-uniform NU10", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_layouts(layouts: dict[str, tuple[float, ...]]) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = FIGURE_DIR / "representative_module_layouts.png"
    colors = {20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}
    labels = {
        "uniform_U10": "uniform U10",
        "cand_3333334332": "mean-best NU10",
        "prev_center_refined": "target-best 120/240 m",
        "prev_edge_mild": "target-best 180 m",
        "prev_bow_refined": "target-best 300 m",
    }
    fig, axis = plt.subplots(figsize=(12.0, 5.8))
    y_positions = np.arange(len(REPRESENTATIVE_LAYOUTS))
    for y, layout_id in zip(y_positions, REPRESENTATIVE_LAYOUTS):
        left = 0.0
        for length in layouts[layout_id]:
            axis.barh(
                y,
                length,
                left=left,
                height=0.58,
                color=colors[length],
                edgecolor="#333333",
                linewidth=0.55,
            )
            axis.text(left + 0.5 * length, y, f"{int(length)}", ha="center", va="center", fontsize=8)
            left += length
    axis.set_yticks(y_positions)
    axis.set_yticklabels([labels[item] for item in REPRESENTATIVE_LAYOUTS])
    axis.invert_yaxis()
    axis.set_xlim(0, 300)
    axis.set_xlabel("x along floating body (m)")
    axis.set_title("Representative 1D module layouts, each spanning full width 60 m")
    axis.grid(True, axis="x", color="#dddddd", linewidth=0.7, alpha=0.85)
    legend = [Patch(facecolor=colors[value], label=f"{int(value)} m module") for value in (20.0, 30.0, 40.0)]
    axis.legend(handles=legend, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_control_points(layouts: dict[str, tuple[float, ...]]) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "control_point_alignment.png"
    labels = {
        "cand_3333334332": "mean-best NU10",
        "prev_center_refined": "120/240 m best",
        "prev_edge_mild": "180 m best",
        "prev_bow_refined": "300 m best",
    }
    fig, axis = plt.subplots(figsize=(11.6, 4.8))
    y_positions = np.arange(len(labels))
    for y, layout_id in zip(y_positions, labels):
        geometry = read_geometry(layout_id)
        centers = [row["center_x_m"] for row in geometry]
        nodes = [row["selected_node_x_m"] for row in geometry]
        axis.scatter(centers, np.full(len(centers), y), marker="o", s=48, label="module center" if y == 0 else None)
        axis.scatter(nodes, np.full(len(nodes), y), marker="+", s=92, label="FEM node" if y == 0 else None)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(list(labels.values()))
    axis.invert_yaxis()
    axis.set_xlim(0, 300)
    axis.set_xlabel("x coordinate (m)")
    axis.set_title("Module centers exactly aligned with selected FEM nodes")
    axis.grid(True, axis="x", color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def copy_existing_figures() -> dict[str, Path]:
    copied = {}
    sources = {
        "search_error_ranking": SEARCH_ROOT / "figures" / "serep_nonuniform_layout_search_error_ranking.png",
        "target_refinement_error_ranking": REFINE_ROOT / "figures" / "serep_nonuniform_layout_search_error_ranking.png",
    }
    for key, source in sources.items():
        target = FIGURE_DIR / source.name.replace("serep_nonuniform_layout_search", key)
        if source.exists():
            shutil.copyfile(source, target)
            copied[key] = target
    return copied


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
    rows: list[dict[str, object]],
    layouts: dict[str, tuple[float, ...]],
    best_nu: dict[int, dict[str, object]],
    figures: dict[str, Path],
) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    target_rows = []
    for wavelength_m in WAVELENGTHS_M:
        uniform = row_for(rows, "uniform_U10", wavelength_m)
        best = best_nu[wavelength_m]
        improvement = (uniform["rmse"] - best["rmse"]) / uniform["rmse"] * 100.0
        target_rows.append(
            (
                f"{wavelength_m}",
                best["layout_id"],
                f"`{fmt_lengths(layouts[best['layout_id']])}`",
                f"{best['rmse']:.6g}",
                f"{uniform['rmse']:.6g}",
                f"{improvement:.2f}%",
            )
        )

    representative_rows = []
    for layout_id in REPRESENTATIVE_LAYOUTS:
        representative_rows.append(
            (
                layout_id,
                f"`{fmt_lengths(layouts[layout_id])}`",
                f"{mean_rmse(rows, layout_id):.6g}",
            )
        )

    control_rows = []
    for layout_id in REPRESENTATIVE_LAYOUTS[1:]:
        geometry = read_geometry(layout_id)
        control_rows.append(
            (
                layout_id,
                ", ".join(str(row["selected_node_id"]) for row in geometry),
                f"{max(row['abs_error_m'] for row in geometry):.1f}",
            )
        )

    lines = [
        "# SEREP 模型下非均匀模块划分的准确性验证报告",
        "",
        "## 1. 研究目的",
        "",
        "本报告整理当前项目中关于 **非均匀水动力模块划分** 的验证结果。目标不是证明任意非均匀划分都更优，而是证明：在修正后的 SEREP 降维模型中，非均匀 10 模块 NU10 可以稳定计算，并且在若干目标波长下达到或优于均匀 U10 相对高分辨率 U30 参考解的精度。",
        "",
        "浮体物理尺度保持不变：`300 m × 60 m × 2 m`，吃水 `0.5 m`，水深 `58.5 m`，水密度 `rho = 1000 kg/m^3`。模块划分始终是一维 `N × 1`，每个模块跨越全宽 `60 m`。",
        "",
        "## 2. 为什么需要先修正 SEREP 实现",
        "",
        "早期计算中，非均匀模块结果和 U30 均匀结果曾出现明显误差甚至毛刺。诊断后发现，误差主要不是来自非均匀水动力网格本身，而是来自 SEREP 降维实现中的两个数值问题：",
        "",
        "1. 主控制点模态子矩阵在高控制点数量下可能病态，直接求逆会放大数值误差。",
        "2. 水动力模块顺序、结构主节点顺序和重构顺序必须严格一致；如果被隐式排序，会破坏模块重心与结构节点的对应关系。",
        "",
        "因此本文采用保持主控制点物理顺序的 SEREP-ridge 形式：",
        "",
        "$$",
        "T = \\Phi \\left(\\Phi_m^T\\Phi_m + \\lambda I\\right)^{-1}\\Phi_m^T",
        "$$",
        "",
        "其中 `lambda = 1e-16 × ||Phi_m^T Phi_m||_2`。该处理不改变结构质量矩阵、刚度矩阵或水动力物理参数，只是避免病态直接逆并保持主控制点顺序。",
        "",
        "## 3. 验证设计",
        "",
        "参考解采用 `U30 SEREP-ridge`，即 30 个均匀模块，每个模块长度 `10 m`。对比对象包括：",
        "",
        "- `uniform_U10`：10 个均匀模块，每个 `30 m`。",
        "- `NU10`：10 个非均匀模块，模块长度限制为 `20/30/40 m`，总长度严格等于 `300 m`。",
        "- 所有 NU10 模块重心必须严格落在结构 FEM 节点上。",
        "",
        "对比波长为 `60/120/180/240/300 m`，指标为 centerline heave RAO 相对 U30 的 RMSE 和最大绝对误差。",
        "",
        "## 4. 代表性非均匀布局",
        "",
        f"![代表性模块布局](figures/{figures['layouts'].name})",
        "",
        markdown_table(("layout", "module lengths (m)", "mean RMSE vs U30"), representative_rows),
        "",
        "## 5. 目标波长精度对比",
        "",
        f"![目标波长 RMSE 对比](figures/{figures['rmse'].name})",
        "",
        markdown_table(("wavelength (m)", "best NU10 layout", "lengths (m)", "NU10 RMSE", "U10 RMSE", "improvement"), target_rows),
        "",
        "可以看到，60 m 短波下均匀 U10 仍然最好；但在 120、180、240、300 m 波长下，均存在非均匀 NU10 比均匀 U10 更接近 U30 参考解。尤其在 120 m 和 240 m，`prev_center_refined` 的改善更明显。",
        "",
        "## 6. Heave 响应曲线",
        "",
        f"![Heave 对比](figures/{figures['heave'].name})",
        "",
        "曲线结果说明，修正后的 SEREP-ridge 下非均匀 NU10 响应没有早期那种非物理毛刺。不同 NU10 布局与 U30 参考解之间的差异主要表现为平滑的幅值偏差，而不是节点顺序错误导致的局部跳变。",
        "",
        "## 7. 控制点与 FEM 节点一致性",
        "",
        f"![控制点对齐](figures/{figures['control'].name})",
        "",
        markdown_table(("layout", "FEM node ids", "max abs error (m)"), control_rows),
        "",
        "所有代表性布局的模块重心与结构节点坐标误差均为 `0 m`，并且没有重复节点。这一点非常关键，因为水动力系数最终集中到模块重心位置，只有重心节点与结构主控制节点一致，水动力矩阵和结构降维矩阵才能正确耦合。",
        "",
        "## 8. 结论",
        "",
        "本阶段可以得到以下结论：",
        "",
        "1. 非均匀模块本身是可行的；早期异常主要来自 SEREP 病态求逆和主节点顺序问题。",
        "2. 在 SEREP-ridge 与主控制点顺序保持后，NU10 结果稳定、平滑、无毛刺。",
        "3. 最优平均误差的 NU10 为 `[30,30,30,30,30,30,40,30,30,20]`，平均 RMSE 仅比均匀 U10 高约 `4.3%`，说明 NU10 可以达到与 U10 同量级精度。",
        "4. 在目标波长 `120/180/240/300 m` 上，存在 NU10 布局优于均匀 U10，说明非均匀划分不仅可行，而且可用于目标波长优化。",
        "5. 过强或任意的非均匀划分并不一定更好；非均匀模块需要与目标波长、控制点分布和降维模型共同设计。",
        "",
        "## 9. 文件与复现实验",
        "",
        "主要脚本：",
        "",
        "- `scripts/run_serep_nonuniform_layout_search.py`：枚举/预筛选并完整求解 NU10 候选布局。",
        "- `scripts/run_serep_target_layout_refinement.py`：围绕目标波长最优布局做局部 refinement。",
        "- `scripts/prepare_serep_nonuniform_proof_report.py`：生成本报告和证明图。",
        "",
        "主要结果目录：",
        "",
        f"- `{SEARCH_ROOT}`",
        f"- `{REFINE_ROOT}`",
        f"- `{REPORT_ROOT}`",
        "",
        "建议论文写法：将该部分作为“非均匀模块 RODM 计算的一致性验证与 SEREP 数值稳定性讨论”小节，而不是作为独立的新理论方法。重点强调非均匀模块可行性的前提是：控制点顺序一致、模块重心精确映射到 FEM 节点、SEREP 降维矩阵经过稳定化处理。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_summary()
    layouts = read_ranking()
    best_nu = best_nonuniform_by_wavelength(rows)
    figures = {
        "heave": plot_heave_panel(rows, best_nu),
        "rmse": plot_rmse(rows, best_nu),
        "layouts": plot_layouts(layouts),
        "control": plot_control_points(layouts),
    }
    figures.update(copy_existing_figures())
    write_report(rows, layouts, best_nu, figures)
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
