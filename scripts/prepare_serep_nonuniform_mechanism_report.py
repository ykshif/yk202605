"""Prepare mechanism-analysis plots and Markdown for non-uniform SEREP results."""

from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_serep_nonuniform_design_study as base  # noqa: E402
import prepare_serep_nonuniform_proof_report as proof  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402


REFINE_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_target_refinement"
REPORT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_mechanism_report"
FIGURE_DIR = REPORT_ROOT / "figures"
TABLE_DIR = REPORT_ROOT / "tables"
REPORT_PATH = REPORT_ROOT / "serep_nonuniform_mechanism_analysis.md"
TARGET_WAVELENGTHS_M = (120, 180, 240, 300)
BODY_LENGTH_M = 300.0


def read_target_best() -> dict[int, str]:
    best: dict[int, str] = {}
    with (REFINE_ROOT / "serep_target_refinement_best_by_wavelength.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            wavelength_m = int(row["wavelength_m"])
            if wavelength_m in TARGET_WAVELENGTHS_M:
                best[wavelength_m] = row["best_layout_id"]
    return best


def response_heave(summary_rows: list[dict[str, object]], layout_id: str, wavelength_m: int) -> tuple[np.ndarray, np.ndarray]:
    row = proof.row_for(summary_rows, layout_id, wavelength_m)
    response = np.load(row["response_path"])
    return extract_centerline_heave(response)


def reference_heave(wavelength_m: int) -> tuple[np.ndarray, np.ndarray]:
    return extract_centerline_heave(np.load(base.reference_response_path(wavelength_m)))


def normalized_curvature(x_over_l: np.ndarray, heave: np.ndarray) -> np.ndarray:
    x_m = x_over_l * BODY_LENGTH_M
    first = np.gradient(heave, x_m)
    second = np.gradient(first, x_m)
    curvature = np.abs(second)
    maximum = float(np.max(curvature))
    if maximum <= 0.0:
        return np.zeros_like(curvature)
    return curvature / maximum


def module_boundaries(layouts: dict[str, tuple[float, ...]], layout_id: str) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(layouts[layout_id], dtype=float))])


def module_average_values(x_m: np.ndarray, values: np.ndarray, boundaries_m: np.ndarray) -> list[float]:
    averages = []
    for start, end in zip(boundaries_m[:-1], boundaries_m[1:]):
        if np.isclose(end, BODY_LENGTH_M):
            mask = (x_m >= start) & (x_m <= end)
        else:
            mask = (x_m >= start) & (x_m < end)
        if np.any(mask):
            averages.append(float(np.mean(values[mask])))
        else:
            center = 0.5 * (start + end)
            averages.append(float(np.interp(center, x_m, values)))
    return averages


def mechanism_rows(
    summary_rows: list[dict[str, object]],
    layouts: dict[str, tuple[float, ...]],
    target_best: dict[int, str],
) -> list[dict[str, object]]:
    rows = []
    for wavelength_m in TARGET_WAVELENGTHS_M:
        best_layout = target_best[wavelength_m]
        x_ref, heave_ref = reference_heave(wavelength_m)
        _, heave_uniform = response_heave(summary_rows, "uniform_U10", wavelength_m)
        _, heave_best = response_heave(summary_rows, best_layout, wavelength_m)
        improvement = np.abs(heave_uniform - heave_ref) - np.abs(heave_best - heave_ref)
        improved = improvement > 0.0
        x_m = x_ref * BODY_LENGTH_M
        max_improvement_index = int(np.argmax(improvement))
        max_deterioration_index = int(np.argmin(improvement))
        boundaries = module_boundaries(layouts, best_layout)
        curvature = normalized_curvature(x_ref, heave_ref)
        module_curvature = module_average_values(x_m, curvature, boundaries)
        rows.append(
            {
                "wavelength_m": wavelength_m,
                "best_layout": best_layout,
                "lengths_m": layouts[best_layout],
                "mean_local_improvement": float(np.mean(improvement)),
                "fraction_nodes_improved": float(np.mean(improved)),
                "max_local_improvement": float(improvement[max_improvement_index]),
                "max_local_improvement_x_m": float(x_m[max_improvement_index]),
                "max_local_deterioration": float(improvement[max_deterioration_index]),
                "max_local_deterioration_x_m": float(x_m[max_deterioration_index]),
                "short_module_mean_curvature": float(
                    np.mean([value for value, length in zip(module_curvature, layouts[best_layout]) if length == 20.0])
                )
                if 20.0 in layouts[best_layout]
                else float("nan"),
                "long_module_mean_curvature": float(
                    np.mean([value for value, length in zip(module_curvature, layouts[best_layout]) if length == 40.0])
                )
                if 40.0 in layouts[best_layout]
                else float("nan"),
            }
        )
    return rows


def write_mechanism_csv(rows: list[dict[str, object]]) -> Path:
    path = TABLE_DIR / "mechanism_summary.csv"
    fieldnames = [
        "wavelength_m",
        "best_layout",
        "lengths_m",
        "mean_local_improvement",
        "fraction_nodes_improved",
        "max_local_improvement",
        "max_local_improvement_x_m",
        "max_local_deterioration",
        "max_local_deterioration_x_m",
        "short_module_mean_curvature",
        "long_module_mean_curvature",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["lengths_m"] = " ".join(str(int(value)) for value in row["lengths_m"])
            writer.writerow(output)
    return path


def write_module_curvature_csv(
    layouts: dict[str, tuple[float, ...]],
    target_best: dict[int, str],
) -> Path:
    path = TABLE_DIR / "module_curvature_by_target_layout.csv"
    fieldnames = [
        "wavelength_m",
        "layout_id",
        "module_id",
        "module_length_m",
        "x_start_m",
        "x_end_m",
        "center_x_m",
        "avg_normalized_curvature",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for wavelength_m, layout_id in target_best.items():
            x_ref, heave_ref = reference_heave(wavelength_m)
            x_m = x_ref * BODY_LENGTH_M
            curvature = normalized_curvature(x_ref, heave_ref)
            boundaries = module_boundaries(layouts, layout_id)
            averages = module_average_values(x_m, curvature, boundaries)
            for index, (start, end, avg) in enumerate(zip(boundaries[:-1], boundaries[1:], averages), start=1):
                writer.writerow(
                    {
                        "wavelength_m": wavelength_m,
                        "layout_id": layout_id,
                        "module_id": index,
                        "module_length_m": int(layouts[layout_id][index - 1]),
                        "x_start_m": start,
                        "x_end_m": end,
                        "center_x_m": 0.5 * (start + end),
                        "avg_normalized_curvature": avg,
                    }
                )
    return path


def plot_error_distribution(
    summary_rows: list[dict[str, object]],
    target_best: dict[int, str],
) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "local_error_distribution_panel.png"
    fig, axes = plt.subplots(len(TARGET_WAVELENGTHS_M), 1, figsize=(11.2, 12.0), sharex=True)
    fig.suptitle("Local heave error distribution: uniform U10 vs target-best NU10", fontsize=15)
    for axis, wavelength_m in zip(axes, TARGET_WAVELENGTHS_M):
        best_layout = target_best[wavelength_m]
        x_ref, heave_ref = reference_heave(wavelength_m)
        _, heave_uniform = response_heave(summary_rows, "uniform_U10", wavelength_m)
        _, heave_best = response_heave(summary_rows, best_layout, wavelength_m)
        x_m = x_ref * BODY_LENGTH_M
        err_uniform = np.abs(heave_uniform - heave_ref)
        err_best = np.abs(heave_best - heave_ref)
        improvement = err_uniform - err_best

        axis.plot(x_m, err_uniform, color="#1f77b4", linewidth=1.5, label="|U10 - U30|")
        axis.plot(x_m, err_best, color="#d62728", linestyle="--", linewidth=1.5, label=f"|NU10 - U30| ({best_layout})")
        axis.fill_between(
            x_m,
            err_uniform,
            err_best,
            where=improvement > 0,
            color="#2ca02c",
            alpha=0.18,
            interpolate=True,
            label="NU10 improves" if wavelength_m == TARGET_WAVELENGTHS_M[0] else None,
        )
        axis.fill_between(
            x_m,
            err_uniform,
            err_best,
            where=improvement <= 0,
            color="#7f7f7f",
            alpha=0.14,
            interpolate=True,
            label="NU10 worsens" if wavelength_m == TARGET_WAVELENGTHS_M[0] else None,
        )
        axis.set_ylabel(f"{wavelength_m} m\nabs error")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="best")
    axes[-1].set_xlabel("x along floating body (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_improvement_heatmap(
    summary_rows: list[dict[str, object]],
    target_best: dict[int, str],
) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "local_improvement_heatmap.png"
    matrix = []
    x_m = None
    for wavelength_m in TARGET_WAVELENGTHS_M:
        best_layout = target_best[wavelength_m]
        x_ref, heave_ref = reference_heave(wavelength_m)
        _, heave_uniform = response_heave(summary_rows, "uniform_U10", wavelength_m)
        _, heave_best = response_heave(summary_rows, best_layout, wavelength_m)
        x_m = x_ref * BODY_LENGTH_M
        matrix.append(np.abs(heave_uniform - heave_ref) - np.abs(heave_best - heave_ref))
    values = np.asarray(matrix)
    vmax = float(np.max(np.abs(values)))

    fig, axis = plt.subplots(figsize=(11.0, 4.8))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(x_m[0]), float(x_m[-1]), len(TARGET_WAVELENGTHS_M) - 0.5, -0.5],
    )
    axis.set_yticks(np.arange(len(TARGET_WAVELENGTHS_M)))
    axis.set_yticklabels([f"{value} m" for value in TARGET_WAVELENGTHS_M])
    axis.set_xlabel("x along floating body (m)")
    axis.set_ylabel("wavelength")
    axis.set_title("Local improvement = |U10 - U30| - |NU10 - U30|")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means NU10 is closer to U30")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_boundary_curvature(
    layouts: dict[str, tuple[float, ...]],
    target_best: dict[int, str],
) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "module_boundaries_vs_u30_curvature.png"
    length_colors = {20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}
    fig, axes = plt.subplots(len(TARGET_WAVELENGTHS_M), 1, figsize=(11.2, 12.0), sharex=True)
    fig.suptitle("Target-best NU10 module boundaries over U30 heave curvature", fontsize=15)
    for axis, wavelength_m in zip(axes, TARGET_WAVELENGTHS_M):
        layout_id = target_best[wavelength_m]
        x_ref, heave_ref = reference_heave(wavelength_m)
        x_m = x_ref * BODY_LENGTH_M
        curvature = normalized_curvature(x_ref, heave_ref)
        boundaries = module_boundaries(layouts, layout_id)
        lengths = layouts[layout_id]

        for start, end, length in zip(boundaries[:-1], boundaries[1:], lengths):
            axis.axvspan(start, end, color=length_colors[length], alpha=0.13)
        for boundary in boundaries:
            axis.axvline(boundary, color="#444444", linewidth=0.5, alpha=0.55)
        axis.plot(x_m, heave_ref, color="#111111", linewidth=1.8, label="U30 heave RAO")
        twin = axis.twinx()
        twin.plot(x_m, curvature, color="#d62728", linewidth=1.2, alpha=0.8, label="normalized curvature")
        twin.set_ylim(0.0, 1.05)
        twin.set_ylabel("curvature")
        axis.set_ylabel(f"{wavelength_m} m\nheave")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
        axis.text(
            0.01,
            0.08,
            layout_id,
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
        )
    axes[0].plot([], [], color="#d62728", linewidth=1.2, label="normalized curvature")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("x along floating body (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_module_curvature(
    layouts: dict[str, tuple[float, ...]],
    target_best: dict[int, str],
) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURE_DIR / "module_length_vs_curvature_panel.png"
    length_colors = {20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}
    fig, axes = plt.subplots(len(TARGET_WAVELENGTHS_M), 1, figsize=(11.2, 10.8), sharex=True)
    fig.suptitle("Average U30 curvature sampled by each target-best NU10 module", fontsize=15)
    for axis, wavelength_m in zip(axes, TARGET_WAVELENGTHS_M):
        layout_id = target_best[wavelength_m]
        x_ref, heave_ref = reference_heave(wavelength_m)
        x_m = x_ref * BODY_LENGTH_M
        curvature = normalized_curvature(x_ref, heave_ref)
        boundaries = module_boundaries(layouts, layout_id)
        centers = 0.5 * (boundaries[:-1] + boundaries[1:])
        averages = module_average_values(x_m, curvature, boundaries)
        lengths = layouts[layout_id]
        axis.bar(
            centers,
            averages,
            width=np.asarray(lengths) * 0.88,
            color=[length_colors[length] for length in lengths],
            edgecolor="#333333",
            linewidth=0.5,
        )
        for center, length, avg in zip(centers, lengths, averages):
            axis.text(center, avg + 0.025, str(int(length)), ha="center", va="bottom", fontsize=8)
        axis.set_ylabel(f"{wavelength_m} m\navg curvature")
        axis.set_ylim(0.0, 1.12)
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
        axis.text(0.01, 0.82, layout_id, transform=axis.transAxes, fontsize=9)
    axes[-1].set_xlabel("x along floating body (m); bar label is module length")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def fmt_lengths(lengths: tuple[float, ...]) -> str:
    return "[" + ", ".join(str(int(value)) for value in lengths) + "]"


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    mechanism: list[dict[str, object]],
    figures: dict[str, Path],
    mechanism_csv: Path,
    module_csv: Path,
) -> None:
    rows = []
    for row in mechanism:
        rows.append(
            (
                row["wavelength_m"],
                row["best_layout"],
                f"`{fmt_lengths(row['lengths_m'])}`",
                f"{row['fraction_nodes_improved'] * 100.0:.1f}%",
                f"{row['max_local_improvement']:.4g} @ {row['max_local_improvement_x_m']:.0f} m",
                f"{row['max_local_deterioration']:.4g} @ {row['max_local_deterioration_x_m']:.0f} m",
            )
        )

    curvature_rows = []
    for row in mechanism:
        curvature_rows.append(
            (
                row["wavelength_m"],
                f"{row['short_module_mean_curvature']:.3f}",
                f"{row['long_module_mean_curvature']:.3f}",
            )
        )

    lines = [
        "# 非均匀模块 SEREP 结果的误差机制分析",
        "",
        "## 1. 分析目的",
        "",
        "前一份验证报告已经说明：在保持主控制点顺序并采用 SEREP-ridge 后，非均匀 NU10 可以稳定进入 RODM，并在若干目标波长下比均匀 U10 更接近 U30 参考解。本报告进一步回答一个更重要的问题：**非均匀布局到底改善了哪里，为什么不是任意非均匀都更好？**",
        "",
        "这里仍只讨论 heave 响应。参考解为 `U30 SEREP-ridge`，比较对象为 `uniform_U10` 与各目标波长下的最优 `NU10`。",
        "",
        "## 2. 局部误差分布",
        "",
        f"![局部误差分布](figures/{figures['error_distribution'].name})",
        "",
        "绿色区域表示 NU10 的局部绝对误差小于均匀 U10，灰色区域表示 NU10 局部误差更大。这个图说明，非均匀模块的优势不是全长均匀发生的，而是在特定波长下改善了某些关键区段，同时可能牺牲另一些区段。",
        "",
        markdown_table(
            (
                "wavelength (m)",
                "best NU10 layout",
                "lengths (m)",
                "nodes improved",
                "max local improvement",
                "max local deterioration",
            ),
            rows,
        ),
        "",
        "## 3. 局部改善热图",
        "",
        f"![局部改善热图](figures/{figures['heatmap'].name})",
        "",
        "热图中的正值表示 NU10 比 U10 更接近 U30。可以看到，120 m 和 240 m 的改善区域更成片，这也是 `prev_center_refined` 在这两个波长上 RMSE 改善更明显的原因。180 m 和 300 m 的改善较弱，说明它们虽然优于 U10，但优势主要来自较窄区域或较小幅度的误差抵消。",
        "",
        "## 4. 模块边界与 U30 响应曲率",
        "",
        f"![模块边界与曲率](figures/{figures['boundary_curvature'].name})",
        "",
        "上图将目标波长最优 NU10 的模块边界叠加在 U30 heave 曲线和归一化曲率上。这里的曲率是沿长度方向 heave RAO 的二阶变化强度，用来近似表示响应空间变化剧烈程度。",
        "",
        f"![模块曲率采样](figures/{figures['module_curvature'].name})",
        "",
        markdown_table(("wavelength (m)", "20 m module avg curvature", "40 m module avg curvature"), curvature_rows),
        "",
        "需要注意，最优布局并不简单等价于“曲率大的地方一定用最短模块”。原因是 RODM 中模块不仅是几何采样单元，也是水动力载荷集中和结构降维控制点的耦合单元。模块长度改变会同时改变水动力积分区域、作用点位置和主控制点分布。因此更准确的说法是：**非均匀模块通过重新分配水动力采样点和控制点位置，改变了局部误差在全长上的分布。**",
        "",
        "## 5. 机制性认识",
        "",
        "从目前结果可以得到以下判断：",
        "",
        "1. 非均匀模块的正确性已经得到支持：响应曲线平滑，控制点精确对齐 FEM 节点，没有节点顺序引起的毛刺。",
        "2. 非均匀模块的优势具有目标波长相关性。120 m 和 240 m 更适合 `prev_center_refined`，300 m 更适合 `prev_bow_refined`，180 m 更适合 `prev_edge_mild`。",
        "3. 平均全波长最优和单一目标波长最优不是同一个问题。平均最优 NU10 `[30,30,30,30,30,30,40,30,30,20]` 非常接近均匀 U10，而目标波长最优布局往往更非均匀。",
        "4. 过强端部加密或随机非均匀不可靠。前面的搜索已经显示，一些看似直观的强非均匀布局完整 RODM 后反而误差更大。",
        "5. 后续如果要提出设计准则，应以目标波长、响应局部误差和结构控制点位置共同决定，而不是只根据模块长度均匀性或水动力直觉。",
        "",
        "## 6. 建议写入论文的表述",
        "",
        "建议将这一部分作为“非均匀模块划分的误差机制与适用性讨论”。可以写成：",
        "",
        "> 非均匀模块划分并不保证在所有波长下均优于均匀划分，但在控制点顺序一致、模块重心与 FEM 节点精确匹配、SEREP 降维矩阵稳定化后，非均匀模块能够获得稳定且物理合理的水弹性响应。其优势主要体现在目标波长下对局部误差分布的重新调节，因此非均匀划分应作为目标工况相关的离散优化问题，而不是随机划分问题。",
        "",
        "## 7. 输出文件",
        "",
        f"- 机制统计表：`{mechanism_csv}`",
        f"- 模块曲率表：`{module_csv}`",
        f"- 图片目录：`{FIGURE_DIR}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = proof.read_summary()
    layouts = proof.read_ranking()
    target_best = read_target_best()
    mechanism = mechanism_rows(summary_rows, layouts, target_best)
    mechanism_csv = write_mechanism_csv(mechanism)
    module_csv = write_module_curvature_csv(layouts, target_best)
    figures = {
        "error_distribution": plot_error_distribution(summary_rows, target_best),
        "heatmap": plot_improvement_heatmap(summary_rows, target_best),
        "boundary_curvature": plot_boundary_curvature(layouts, target_best),
        "module_curvature": plot_module_curvature(layouts, target_best),
    }
    write_report(mechanism, figures, mechanism_csv, module_csv)
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    print(f"tables={TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
