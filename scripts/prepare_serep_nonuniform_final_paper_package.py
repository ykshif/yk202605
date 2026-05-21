"""Collect the SEREP non-uniform module study into a final paper package."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

SWEEP_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_wavelength_sweep"
PAPER_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_paper_section"
LAMBDA_ROOT = REPO_ROOT / "results" / "serep_ridge_lambda_sensitivity"
FINAL_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_final_validation"
MECHANISM_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_mechanism_report"

OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_final_paper_package"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_PATH = OUTPUT_ROOT / "serep_nonuniform_final_paper_package.md"

FIGURE_SOURCES = {
    "control_points": FINAL_ROOT / "figures" / "final_validation_layouts_control_points.png",
    "node_sequence": FINAL_ROOT / "figures" / "final_validation_node_sequence.png",
    "wavelength_heatmap": SWEEP_ROOT / "figures" / "wavelength_sweep_applicability_heatmap.png",
    "wavelength_rmse": SWEEP_ROOT / "figures" / "wavelength_sweep_rmse_curves.png",
    "heave_grid": SWEEP_ROOT / "figures" / "wavelength_sweep_heave_grid.png",
    "weighted_scores": PAPER_ROOT / "figures" / "wavelength_band_weighted_scores.png",
    "local_error": MECHANISM_ROOT / "figures" / "local_error_distribution_panel.png",
    "curvature_boundaries": MECHANISM_ROOT / "figures" / "module_boundaries_vs_u30_curvature.png",
    "lambda_rmse": LAMBDA_ROOT / "figures" / "lambda_sensitivity_rmse_vs_lambda.png",
    "lambda_best_map": LAMBDA_ROOT / "figures" / "lambda_sensitivity_best_layout_map.png",
    "lambda_drift": LAMBDA_ROOT / "figures" / "lambda_sensitivity_response_drift.png",
}

TABLE_SOURCES = {
    "wavelength_metrics": SWEEP_ROOT / "tables" / "wavelength_sweep_by_wavelength.csv",
    "best_by_wavelength": SWEEP_ROOT / "tables" / "wavelength_sweep_best_by_wavelength.csv",
    "layout_summary": SWEEP_ROOT / "tables" / "wavelength_sweep_layout_summary.csv",
    "weighted_summary": PAPER_ROOT / "tables" / "wavelength_band_weighted_summary.csv",
    "best_bands": PAPER_ROOT / "tables" / "wavelength_band_best_layouts.csv",
    "lambda_metrics": LAMBDA_ROOT / "tables" / "lambda_sensitivity_by_wavelength.csv",
    "lambda_drift": LAMBDA_ROOT / "tables" / "lambda_response_drift_vs_1e-16.csv",
    "lambda_master_nodes": LAMBDA_ROOT / "tables" / "lambda_sensitivity_master_nodes.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_existing(sources: dict[str, Path], target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for key, source in sources.items():
        if source.exists():
            target = target_dir / source.name
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


def best_wavelength_rows() -> list[tuple[object, ...]]:
    rows = read_csv(TABLE_SOURCES["best_by_wavelength"])
    return [
        (
            row["wavelength_m"],
            row["best_display_name"],
            f"{float(row['best_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in rows
    ]


def selected_band_rows() -> list[tuple[object, ...]]:
    rows = read_csv(TABLE_SOURCES["best_bands"])
    keep = {
        "full_60_300",
        "short_60_120",
        "long_210_300",
        "target_center_120_150",
        "target_edge_180",
        "target_center_240_270",
        "target_bow_300",
    }
    return [
        (
            row["band_name"],
            row["wavelengths_m"],
            row["best_display_name"],
            f"{float(row['best_weighted_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_weighted_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in rows
        if row["band_id"] in keep
    ]


def lambda_baseline_rows() -> list[tuple[object, ...]]:
    rows = [
        row
        for row in read_csv(TABLE_SOURCES["lambda_metrics"])
        if row["lambda_label"] == "1e-16" and row["layout_id"] != "U30_reference"
    ]
    output = []
    for row in rows:
        if float(row["improvement_vs_same_lambda_U10_percent"]) > 0.0 or row["layout_id"] == "uniform_U10":
            output.append(
                (
                    row["wavelength_m"],
                    row["display_name"],
                    f"{float(row['rmse_vs_same_lambda_U30']):.6g}",
                    f"{float(row['improvement_vs_same_lambda_U10_percent']):.2f}%",
                )
            )
    return output


def lambda_drift_rows() -> list[tuple[object, ...]]:
    rows = read_csv(TABLE_SOURCES["lambda_drift"])
    output = []
    for layout_id in ["U30_reference", "uniform_U10", "NU10_center_refined", "NU10_edge_mild", "NU10_bow_refined"]:
        layout_rows = [row for row in rows if row["layout_id"] == layout_id]
        if not layout_rows:
            continue
        output.append(
            (
                layout_rows[0]["display_name"],
                f"{max(float(row['rmse_vs_lambda_1e-16']) for row in layout_rows):.3e}",
                f"{np.mean([float(row['rmse_vs_lambda_1e-16']) for row in layout_rows]):.3e}",
            )
        )
    return output


def write_report(figures: dict[str, Path], tables: dict[str, Path]) -> None:
    lines = [
        "# SEREP-ridge 非均匀水动力模块研究：最终图文包",
        "",
        "## 1. 当前结论",
        "",
        "现在可以进入论文层面的整理，但表述要克制：非均匀 NU10 不是在全波段无条件优于均匀 U10，而是在控制点顺序、模块重心与 FEM 节点严格对齐，并采用 SEREP-ridge 稳定化之后，表现为一种目标波长相关的离散策略。",
        "",
        "浮体尺寸保持为 `300 m x 60 m x 2 m`，吃水 `0.5 m`，水深 `58.5 m`，水密度 `1000 kg/m^3`。所有水动力模块均为沿长度方向的一维 `N x 1` 划分，每个模块跨越全宽 `60 m`。",
        "",
        "## 2. 控制点与节点顺序",
        "",
        f"![Control points](figures/{figures['control_points'].name})",
        "",
        f"![Node sequence](figures/{figures['node_sequence'].name})",
        "",
        "这两张图建议放在方法验证部分，用来说明非均匀模块不是二维划分，模块重心位置与 FEM 主控制节点一一对应，且节点顺序没有反转或错配。",
        "",
        "## 3. SEREP-ridge 参数敏感性",
        "",
        f"![Lambda RMSE](figures/{figures['lambda_rmse'].name})",
        "",
        f"![Lambda drift](figures/{figures['lambda_drift'].name})",
        "",
        "参数扫描表明，`lambda=1e-16` 不是可以随意替换的数值细节。过大或过小的正则化会使 U30 参考解自身发生明显漂移，因此论文中应明确给出 SEREP-ridge 的参数，并把宽范围参数敏感性作为附录或讨论材料。",
        "",
        markdown_table(("case", "max drift vs 1e-16", "mean drift vs 1e-16"), lambda_drift_rows()),
        "",
        "在固定 `lambda=1e-16` 后，目标波长处的 10 模块最优布局如下：",
        "",
        markdown_table(("wavelength (m)", "case", "RMSE vs U30", "improvement vs U10"), lambda_baseline_rows()),
        "",
        "## 4. 加密波长扫描与适用性",
        "",
        f"![Applicability heatmap](figures/{figures['wavelength_heatmap'].name})",
        "",
        f"![RMSE curves](figures/{figures['wavelength_rmse'].name})",
        "",
        markdown_table(("wavelength (m)", "best case", "best RMSE", "U10 RMSE", "improvement"), best_wavelength_rows()),
        "",
        f"![Heave curves](figures/{figures['heave_grid'].name})",
        "",
        "这组图是证明非均匀模块正确性的核心证据之一：曲线没有早期那类毛刺，且优势只出现在特定波长窗口。建议正文中强调 `120/150/240/270 m` 中部加密更有效、`180 m` 两端轻度加密更有效、`300 m` 迎浪端加密更有效。",
        "",
        "## 5. 目标海况或波长带加权",
        "",
        f"![Weighted bands](figures/{figures['weighted_scores'].name})",
        "",
        markdown_table(("band", "wavelengths (m)", "best case", "best J", "U10 J", "improvement"), selected_band_rows()),
        "",
        "全波段或宽波段平均时，均匀 U10 仍然是更稳健的工程基线；但在较窄的目标波长窗口，非均匀 NU10 可以获得更小的加权 RMSE。这是后续论文要继续深化的方向：用实际海况谱或目标工况概率来定义权重 `w_i`。",
        "",
        "## 6. 误差分布与机理解释",
        "",
        f"![Local error](figures/{figures['local_error'].name})",
        "",
        f"![Curvature boundaries](figures/{figures['curvature_boundaries'].name})",
        "",
        "非均匀划分的效果不应简单解释为哪里曲率大哪里就必须放短模块。更准确的说法是：模块长度同时改变水动力积分区域、载荷集中点和结构主控制点位置，因此是在目标波长下重新分配 RODM 近似误差。",
        "",
        "## 7. 建议写入论文的结构",
        "",
        "1. 方法部分：说明一维非均匀模块划分、模块重心到 FEM 主控制节点映射、SEREP-ridge 降维和 `lambda=1e-16`。",
        "2. 程序验证部分：放控制点顺序图、严格均匀退化、U30 参考解说明。",
        "3. 结果部分：放波长适用性热图、heave 响应曲线、逐波长最优布局表。",
        "4. 讨论部分：放目标波长带加权、误差分布机理、参数敏感性。",
        "5. 结论部分：强调非均匀模块是目标工况相关策略，不是全局替代均匀划分。",
        "",
        "## 8. 文件索引",
        "",
        f"- 图文包目录：`{OUTPUT_ROOT}`",
        f"- 图目录：`{FIGURE_DIR}`",
        f"- 表目录：`{TABLE_DIR}`",
        f"- 参数敏感性脚本：`{REPO_ROOT / 'scripts' / 'run_serep_ridge_lambda_sensitivity.py'}`",
        f"- 本图文包脚本：`{REPO_ROOT / 'scripts' / 'prepare_serep_nonuniform_final_paper_package.py'}`",
        "",
        "复制到本图文包的关键表格：",
    ]
    for key, path in tables.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def write_manifest(figures: dict[str, Path], tables: dict[str, Path]) -> Path:
    path = OUTPUT_ROOT / "final_paper_package_manifest.json"
    payload = {
        "report": str(REPORT_PATH),
        "figures": {key: str(value) for key, value in figures.items()},
        "tables": {key: str(value) for key, value in tables.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    figures = copy_existing(FIGURE_SOURCES, FIGURE_DIR)
    tables = copy_existing(TABLE_SOURCES, TABLE_DIR)
    missing_figures = sorted(set(FIGURE_SOURCES) - set(figures))
    missing_tables = sorted(set(TABLE_SOURCES) - set(tables))
    if missing_figures or missing_tables:
        raise FileNotFoundError(
            f"missing figures={missing_figures}, missing tables={missing_tables}; "
            "run the wavelength sweep, lambda sensitivity, and paper-section scripts first"
        )

    write_report(figures, tables)
    manifest = write_manifest(figures, tables)
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    print(f"tables={TABLE_DIR}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
