"""Prepare a paper-style section for the SEREP non-uniform module study."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]

SWEEP_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_wavelength_sweep"
FINAL_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_final_validation"
MECHANISM_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_mechanism_report"
OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_paper_section"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "paper_section_nonuniform_serep.md"

METRICS_CSV = SWEEP_ROOT / "tables" / "wavelength_sweep_by_wavelength.csv"
LAYOUT_SUMMARY_CSV = SWEEP_ROOT / "tables" / "wavelength_sweep_layout_summary.csv"
BEST_BY_WAVELENGTH_CSV = SWEEP_ROOT / "tables" / "wavelength_sweep_best_by_wavelength.csv"

LAYOUT_ORDER = (
    "uniform_U10",
    "NU10_center_refined",
    "NU10_edge_mild",
    "NU10_bow_refined",
    "NU10_mean_best",
)

LAYOUT_LABELS = {
    "uniform_U10": "U10 uniform",
    "NU10_center_refined": "NU10 center-refined",
    "NU10_edge_mild": "NU10 edge-mild",
    "NU10_bow_refined": "NU10 bow-refined",
    "NU10_mean_best": "NU10 mean-best",
}

COLORS = {
    "uniform_U10": "#1f77b4",
    "NU10_center_refined": "#ff7f0e",
    "NU10_edge_mild": "#9467bd",
    "NU10_bow_refined": "#d62728",
    "NU10_mean_best": "#2ca02c",
}

BANDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "short_60_120": ("short-wave band", (60, 90, 120)),
    "transition_120_180": ("transition band", (120, 150, 180)),
    "middle_120_210": ("middle-wave band", (120, 150, 180, 210)),
    "long_210_300": ("long-wave band", (210, 240, 270, 300)),
    "improvement_targets": ("target-improvement points", (120, 150, 180, 240, 270, 300)),
    "target_center_120_150": ("center target band", (120, 150)),
    "target_edge_180": ("edge target wavelength", (180,)),
    "target_center_240_270": ("center long target band", (240, 270)),
    "target_bow_300": ("bow target wavelength", (300,)),
    "full_60_300": ("full sweep", (60, 90, 120, 150, 180, 210, 240, 270, 300)),
}

FIGURE_SOURCES = {
    "applicability_heatmap": SWEEP_ROOT / "figures" / "wavelength_sweep_applicability_heatmap.png",
    "heave_grid": SWEEP_ROOT / "figures" / "wavelength_sweep_heave_grid.png",
    "rmse_curves": SWEEP_ROOT / "figures" / "wavelength_sweep_rmse_curves.png",
    "control_points": FINAL_ROOT / "figures" / "final_validation_layouts_control_points.png",
    "node_sequence": FINAL_ROOT / "figures" / "final_validation_node_sequence.png",
    "local_error_distribution": MECHANISM_ROOT / "figures" / "local_error_distribution_panel.png",
    "curvature_boundaries": MECHANISM_ROOT / "figures" / "module_boundaries_vs_u30_curvature.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(METRICS_CSV):
        rows.append(
            {
                "layout_id": row["layout_id"],
                "display_name": row["display_name"],
                "category": row["category"],
                "wavelength_m": int(row["wavelength_m"]),
                "rmse_vs_U30": float(row["rmse_vs_U30"]),
                "max_abs_vs_U30": float(row["max_abs_vs_U30"]),
                "roughness": float(row["roughness"]),
                "improvement_vs_U10_percent": float(row["improvement_vs_U10_percent"]),
            }
        )
    return rows


def value_for(rows: list[dict[str, object]], layout_id: str, wavelength_m: int, key: str) -> float:
    for row in rows:
        if row["layout_id"] == layout_id and row["wavelength_m"] == wavelength_m:
            return float(row[key])
    raise KeyError((layout_id, wavelength_m, key))


def write_weighted_band_tables(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    summary_path = TABLE_DIR / "wavelength_band_weighted_summary.csv"
    best_path = TABLE_DIR / "wavelength_band_best_layouts.csv"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    best_rows = []
    for band_id, (band_name, wavelengths) in BANDS.items():
        u10_score = float(np.mean([value_for(rows, "uniform_U10", wl, "rmse_vs_U30") for wl in wavelengths]))
        scored = []
        for layout_id in LAYOUT_ORDER:
            score = float(np.mean([value_for(rows, layout_id, wl, "rmse_vs_U30") for wl in wavelengths]))
            max_abs = float(np.mean([value_for(rows, layout_id, wl, "max_abs_vs_U30") for wl in wavelengths]))
            roughness = float(np.mean([value_for(rows, layout_id, wl, "roughness") for wl in wavelengths]))
            improvement = (u10_score - score) / u10_score * 100.0 if u10_score else 0.0
            row = {
                "band_id": band_id,
                "band_name": band_name,
                "wavelengths_m": " ".join(str(value) for value in wavelengths),
                "layout_id": layout_id,
                "display_name": LAYOUT_LABELS[layout_id],
                "weighted_rmse_vs_U30": score,
                "weighted_max_abs_vs_U30": max_abs,
                "weighted_roughness": roughness,
                "improvement_vs_U10_percent": improvement,
            }
            summary_rows.append(row)
            scored.append(row)
        best = min(scored, key=lambda item: item["weighted_rmse_vs_U30"])
        best_rows.append(
            {
                "band_id": band_id,
                "band_name": band_name,
                "wavelengths_m": " ".join(str(value) for value in wavelengths),
                "best_layout_id": best["layout_id"],
                "best_display_name": best["display_name"],
                "best_weighted_rmse_vs_U30": best["weighted_rmse_vs_U30"],
                "uniform_U10_weighted_rmse_vs_U30": u10_score,
                "best_improvement_vs_U10_percent": best["improvement_vs_U10_percent"],
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with best_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    return summary_path, best_path


def read_weighted_summary(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def plot_band_scores(summary_path: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_weighted_summary(summary_path)
    band_ids = list(BANDS)
    band_labels = [BANDS[band_id][0] for band_id in band_ids]
    x = np.arange(len(band_ids))
    width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(18.4, 6.2))
    offsets = np.linspace(-2, 2, len(LAYOUT_ORDER)) * width
    for offset, layout_id in zip(offsets, LAYOUT_ORDER):
        values = [
            float(next(row for row in rows if row["band_id"] == band_id and row["layout_id"] == layout_id)["weighted_rmse_vs_U30"])
            for band_id in band_ids
        ]
        axes[0].bar(x + offset, values, width, color=COLORS[layout_id], label=LAYOUT_LABELS[layout_id])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_labels, rotation=32, ha="right")
    axes[0].set_ylabel("weighted RMSE vs U30")
    axes[0].set_title("Band-weighted heave accuracy")
    axes[0].grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, fontsize=8)

    plot_layouts = [layout_id for layout_id in LAYOUT_ORDER if layout_id != "uniform_U10"]
    values = np.asarray(
        [
            [
                float(
                    next(
                        row
                        for row in rows
                        if row["band_id"] == band_id and row["layout_id"] == layout_id
                    )["improvement_vs_U10_percent"]
                )
                for band_id in band_ids
            ]
            for layout_id in plot_layouts
        ]
    )
    vmax = max(1.0, float(np.max(np.abs(values))))
    image = axes[1].imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(np.arange(len(band_ids)))
    axes[1].set_xticklabels(band_labels, rotation=32, ha="right")
    axes[1].set_yticks(np.arange(len(plot_layouts)))
    axes[1].set_yticklabels([LAYOUT_LABELS[item] for item in plot_layouts])
    axes[1].set_title("Band improvement vs U10 (%)")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axes[1].text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    cbar = fig.colorbar(image, ax=axes[1])
    cbar.set_label("positive means lower weighted RMSE than U10")

    fig.suptitle("Wavelength-band weighted evaluation of NU10 layouts", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURE_DIR / "wavelength_band_weighted_scores.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def copy_key_figures() -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, source in FIGURE_SOURCES.items():
        if source.exists():
            target = FIGURE_DIR / source.name
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


def fmt_lengths(text: str) -> str:
    return "[" + ", ".join(value for value in text.split()) + "]"


def write_report(
    *,
    weighted_summary_path: Path,
    best_band_path: Path,
    band_figure: Path,
    copied_figures: dict[str, Path],
) -> None:
    layout_summary = read_csv(LAYOUT_SUMMARY_CSV)
    best_by_wavelength = read_csv(BEST_BY_WAVELENGTH_CSV)
    band_best = read_csv(best_band_path)

    layout_rows = []
    for row in layout_summary:
        if row["layout_id"] == "U30_reference":
            continue
        layout_rows.append(
            (
                row["display_name"],
                f"`{fmt_lengths(row['module_lengths_m'])}`",
                f"{float(row['mean_rmse_vs_U30']):.6g}",
                f"{float(row['mean_improvement_vs_U10_percent']):.2f}%",
                row["better_than_U10_count"],
                row["max_control_point_abs_error_m"],
            )
        )

    wavelength_rows = [
        (
            row["wavelength_m"],
            row["best_display_name"],
            f"{float(row['best_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in best_by_wavelength
    ]

    band_rows = [
        (
            row["band_name"],
            f"`{row['wavelengths_m']}`",
            row["best_display_name"],
            f"{float(row['best_weighted_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_weighted_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in band_best
    ]

    lines = [
        "# 论文结果章节草稿：SEREP-ridge 框架下非均匀水动力模块划分",
        "",
        "## 1. 研究问题与章节定位",
        "",
        "本节讨论大型浮体 RODM 水弹性计算中水动力模块沿长度方向的非均匀划分问题。浮体总尺寸保持为 `300 m x 60 m x 2 m`，吃水为 `0.5 m`，水深为 `58.5 m`，水密度取 `1000 kg/m^3`。所有水动力模块均采用一维 `N x 1` 布局，每个模块跨越全宽 `60 m`，不进行宽度方向划分。",
        "",
        "本节的核心命题不是证明任意非均匀划分均优于均匀划分，而是证明：在主控制点顺序一致、模块重心精确映射到 FEM 节点、并采用 SEREP-ridge 稳定化降维之后，非均匀 NU10 可以作为一种目标波长相关的水动力离散策略，在部分波长区间内比均匀 U10 更接近高分辨率 U30 参考解。",
        "",
        "## 2. SEREP 稳定化与控制点一致性",
        "",
        "早期计算中出现的非物理毛刺主要来自两个数值问题：一是传统 SEREP 中主控制点模态子矩阵直接求逆可能病态；二是水动力模块顺序、结构主节点顺序和响应重构顺序若不一致，会破坏模块重心与结构节点的对应关系。为此本文采用保持主节点物理顺序的 SEREP-ridge 形式：",
        "",
        "$$",
        "T = \\Phi \\left(\\Phi_m^T\\Phi_m + \\lambda I\\right)^{-1}\\Phi_m^T",
        "$$",
        "",
        "其中 `lambda = 1e-16 x ||Phi_m^T Phi_m||_2`。该处理不改变质量矩阵、刚度矩阵或水动力物理参数，只用于避免病态逆并保持主控制点顺序。",
        "",
        f"![控制点一致性](figures/{copied_figures['control_points'].name})",
        "",
        f"![节点顺序](figures/{copied_figures['node_sequence'].name})",
        "",
        "最终验证案例中，所有模块重心均精确落在 FEM 主控制节点上，最大坐标误差为 `0 m`，且无重复主节点。因此后续差异可以解释为模块离散策略本身的影响，而不是节点映射错误。",
        "",
        "## 3. 参考解、基准解与非均匀布局",
        "",
        "参考解采用 `U30 SEREP-ridge`，即 30 个均匀模块，每个模块长 `10 m`。工程基准采用 `U10`，即 10 个均匀模块，每个模块长 `30 m`。非均匀 NU10 保持总模块数为 10，总长度为 300 m，模块长度限制在 `20/30/40 m`，保证模块重心位于 5 m 结构网格节点上。",
        "",
        markdown_table(
            ("case", "module lengths (m)", "mean RMSE", "mean improvement", "better count", "max node error"),
            layout_rows,
        ),
        "",
        "全波段平均意义上，均匀 U10 仍然是最稳健的 10 模块基准。非均匀布局的价值不应被表述为全波段无条件更优，而应表述为在特定目标波长下具有精度优势。",
        "",
        "## 4. 加密波长扫描与适用性地图",
        "",
        "为避免只在少数典型波长上挑选结果，本文将波长集合扩展为 `60, 90, 120, 150, 180, 210, 240, 270, 300 m`，并固定代表性布局进行扫描。",
        "",
        f"![适用性热图](figures/{copied_figures['applicability_heatmap'].name})",
        "",
        markdown_table(
            ("wavelength (m)", "best case", "best RMSE", "U10 RMSE", "improvement"),
            wavelength_rows,
        ),
        "",
        "结果显示，`60 m` 和 `90 m` 短波下均匀 U10 最优；`120 m`、`150 m`、`240 m` 和 `270 m` 下中部加密布局更接近 U30；`180 m` 下两端轻度加密布局更优；`300 m` 长波下迎浪端加密布局更优。该规律支持“目标波长相关离散策略”的解释。",
        "",
        f"![RMSE 曲线](figures/{copied_figures['rmse_curves'].name})",
        "",
        f"![Heave 响应曲线](figures/{copied_figures['heave_grid'].name})",
        "",
        "响应曲线在所有加密波长下保持平滑，没有出现早期由节点顺序或 SEREP 病态逆引起的局部毛刺。这进一步说明当前 NU10 结果具有数值一致性。",
        "",
        "## 5. 波长带加权评价",
        "",
        "为了从单一规则波推广到目标波长带，可定义加权误差指标：",
        "",
        "$$",
        "J = \\sum_i w_i \\mathrm{RMSE}(\\lambda_i)",
        "$$",
        "",
        "本文先采用等权重评价短波、中波、长波及全波段表现。若后续给定实际海况谱，可将 `w_i` 替换为谱能量或目标工况概率权重。",
        "",
        f"![波长带加权评价](figures/{band_figure.name})",
        "",
        markdown_table(
            ("band", "wavelengths (m)", "best case", "best J", "U10 J", "improvement"),
            band_rows,
        ),
        "",
        "波长带加权结果进一步说明：若以全波段或宽波段等权平均为目标，均匀 U10 仍然最稳；若目标集中在较窄的波长窗口，例如 `120-150 m`、`180 m`、`240-270 m` 或 `300 m`，则对应的非均匀布局可以取得优势。因此，非均匀模块更适合被定义为目标工况相关的离散优化，而不是替代均匀划分的通用规则。",
        "",
        "## 6. 误差分布机理解释",
        "",
        f"![局部误差分布](figures/{copied_figures['local_error_distribution'].name})",
        "",
        f"![模块边界与曲率](figures/{copied_figures['curvature_boundaries'].name})",
        "",
        "局部误差分布表明，非均匀模块的改善并非沿全长均匀发生，而是在特定波长下改变了局部误差的空间分布。模块长度同时影响水动力积分区域、载荷集中位置和结构主控制点布置，因此最优非均匀布局不能简单理解为“曲率大的地方一定使用最短模块”。更准确的表述是：非均匀划分通过重新分配水动力采样点和控制点位置，调节了 RODM 对目标波长响应形态的近似误差。",
        "",
        "## 7. 本节结论",
        "",
        "1. 非均匀水动力模块生成与 SEREP-ridge RODM 耦合已经通过几何、控制点、节点顺序和响应曲线一致性验证。",
        "2. U30 SEREP-ridge 可作为本文后续比较的高分辨率参考解，U10 均匀划分可作为工程基准。",
        "3. NU10 的优势具有目标波长相关性：短波 `60/90 m` 下 U10 更稳；`120/150/240/270 m` 中部加密更有效；`180 m` 两端轻度加密更有效；`300 m` 迎浪端加密更有效。",
        "4. 宽波段平均下 U10 仍然表现稳健，因此论文中不应宣称非均匀划分全局优于均匀划分。",
        "5. 更合理的论文贡献表述是：提出并验证了一种目标工况相关的非均匀水动力模块离散策略，并指出其成立条件为主控制点顺序一致、模块重心与 FEM 节点精确对齐、SEREP 降维矩阵稳定化。",
        "",
        "## 8. 建议写入论文的精简表述",
        "",
        "> The non-uniform hydrodynamic module division should be interpreted as a target-wavelength-dependent discretization strategy rather than a universally superior replacement of uniform division. With the ordered SEREP-ridge reduction and exact alignment between module centers and FEM master nodes, the NU10 models produce smooth and physically consistent heave responses. The extended wavelength sweep shows that selected non-uniform layouts improve the accuracy at specific wavelengths, whereas the uniform U10 model remains robust for broad-band averaged performance.",
        "",
        "## 9. 数据与脚本",
        "",
        f"- 章节生成脚本：`{SCRIPT_DIR / 'prepare_serep_nonuniform_paper_section.py'}`",
        f"- 波长带加权总表：`{weighted_summary_path}`",
        f"- 波长带最优布局表：`{best_band_path}`",
        f"- 本章节目录：`{OUTPUT_ROOT}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def write_manifest(outputs: dict[str, Path]) -> Path:
    path = OUTPUT_ROOT / "paper_section_manifest.json"
    path.write_text(
        json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = metric_rows()
    weighted_summary_path, best_band_path = write_weighted_band_tables(rows)
    band_figure = plot_band_scores(weighted_summary_path)
    copied_figures = copy_key_figures()
    write_report(
        weighted_summary_path=weighted_summary_path,
        best_band_path=best_band_path,
        band_figure=band_figure,
        copied_figures=copied_figures,
    )
    manifest = write_manifest(
        {
            "report": REPORT_PATH,
            "weighted_summary": weighted_summary_path,
            "best_band": best_band_path,
            "band_figure": band_figure,
            **{f"figure_{key}": value for key, value in copied_figures.items()},
        }
    )
    print(f"report={REPORT_PATH}")
    print(f"tables={TABLE_DIR}")
    print(f"figures={FIGURE_DIR}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
