"""Build a unified continuous-body and hinge hydroelastic validation report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "hydroelastic_validation"
REPORT_PATH = RESULT_ROOT / "hydroelastic_validation_report.md"
DOC_REPORT_PATH = REPO_ROOT / "docs" / "hydroelastic_validation_report.md"
REGULAR_INDEX_PATH = REPO_ROOT / "results" / "regular_wave_batch" / "figure_index.json"
REGULAR_REPORT_PATH = REPO_ROOT / "results" / "regular_wave_batch" / "regular_wave_batch_validation_report.md"
REGULAR_300_DIAGNOSTIC_PATH = REPO_ROOT / "docs" / "regular_wave_300m_diagnostic_report.md"
HINGE_METRICS_PATH = REPO_ROOT / "results" / "yoon_hinge_standard" / "metrics.json"
HINGE_REPORT_PATH = REPO_ROOT / "results" / "yoon_hinge_standard" / "report.md"


def load_json(path: Path) -> dict[str, object]:
    """Load JSON or return an empty mapping if it is not available."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def path_exists_text(path: str | Path | None) -> str:
    """Return a Chinese status string for a path."""

    if not path:
        return "未生成"
    return "已生成" if Path(path).exists() else "缺失"


def first_existing(paths: list[str]) -> str:
    """Return the first existing path string from a list."""

    for path in paths:
        if path and Path(path).exists():
            return path
    return paths[0] if paths else ""


def hinge_case_figure(case_id: str, item: dict[str, object]) -> str:
    """Pick a representative hinge comparison figure for the report."""

    panels = [str(path) for path in item.get("comparison_panels", [])]
    figures = [str(path) for path in item.get("figures", [])]
    legacy = [str(path) for path in item.get("legacy_figures", [])]

    if "single" in case_id:
        return first_existing(panels + figures + legacy)
    centerline = [path for path in panels + figures if "centerline" in path]
    return first_existing(centerline + panels + figures + legacy)


def hinge_case_type(case_id: str) -> str:
    """Return a user-facing hinge case category."""

    if case_id.startswith("single"):
        return "单铰接"
    if case_id.startswith("double"):
        return "双铰接"
    return "铰接"


def hinge_reference_summary(item: dict[str, object]) -> str:
    """Summarize whether digitized reference/experiment curves are configured."""

    manifest = item.get("manifest", {})
    lines = manifest.get("comparison_lines", []) if isinstance(manifest, dict) else []
    reference_count = 0
    experiment_count = 0
    legacy_count = 0
    for line in lines:
        reference_count += len(line.get("reference_curves", []))
        experiment_count += len(line.get("experiment_curves", []))
        legacy_count += len(line.get("legacy_figure_paths", []))

    parts = []
    if reference_count:
        parts.append(f"他人数值曲线 {reference_count} 条")
    if experiment_count:
        parts.append(f"实验点 {experiment_count} 组")
    if legacy_count:
        parts.append(f"历史论文图 {legacy_count} 张")
    return "，".join(parts) if parts else "当前仅输出 RODM 曲线"


def write_report(regular_index: dict[str, object], hinge_metrics: dict[str, object]) -> Path:
    """Write the unified readable Markdown report."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    regular_cases = regular_index.get("cases", [])
    hinge_cases = hinge_metrics.get("cases", {})
    panel_png = str(regular_index.get("panel_png", ""))

    lines: list[str] = [
        "# 连续性浮体与铰接浮体水弹性统一验证报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 报告目标",
        "",
        "本报告把当前标准化代码的两条核心验证线合并到同一份说明中：",
        "",
        "- 连续性浮体：300 m x 60 m 连续体浮体，规则波波长 60 m、120 m、180 m、240 m、300 m。",
        "- 铰接浮体：单铰接约束和双铰接约束，并与实验结果、他人数值结果或历史论文图件进行对比。",
        "",
        "报告侧重可读性和图件索引，不在正文中展开 RMSE 等误差表。数值算法以当前已验证的标准脚本为准。",
        "",
        "## 2. 代码入口",
        "",
        "| 验证对象 | 标准脚本 | 结果目录 |",
        "| --- | --- | --- |",
        f"| 连续性浮体 60-300 m | `{REPO_ROOT / 'scripts' / 'run_regular_wave_batch_validation.py'}` | `{REPO_ROOT / 'results' / 'regular_wave_batch'}` |",
        f"| 单铰/双铰浮体 | `{REPO_ROOT / 'scripts' / 'run_yoon_hinge_cases.py'}` | `{REPO_ROOT / 'results' / 'yoon_hinge_standard'}` |",
        f"| 本统一报告 | `{REPO_ROOT / 'scripts' / 'build_hydroelastic_validation_report.py'}` | `{RESULT_ROOT}` |",
        "",
        "## 3. 连续性浮体规则波验证",
        "",
        "连续性浮体计算采用 RODM 频域水弹性流程：读取结构质量/刚度矩阵和 Capytaine 水动力数据，删除每节点第 6 自由度，选取 10 个主节点降阶，求解频域动力方程，并提取中心线 heave RAO 与实验/他人数值结果对比。",
        "",
    ]

    if panel_png and Path(panel_png).exists():
        lines.extend(
            [
                "### 3.1 五个波长汇总图",
                "",
                f"![连续性浮体五个波长汇总]({panel_png})",
                "",
            ]
        )

    lines.extend(
        [
            "### 3.2 分波长结果索引",
            "",
            "| 波长 (m) | 响应状态 | 图件状态 | 300 m 方向修正 | 图件 |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in regular_cases:
        figure = str(item.get("figure_png", ""))
        reverse = "是" if item.get("reverse_hydrodynamic_node_order") else "否"
        lines.append(
            f"| {item.get('wavelength_m')} | {item.get('response_status')} | "
            f"{item.get('figure_status')} | {reverse} | `{figure}` |"
        )

    lines.extend(
        [
            "",
            "### 3.3 连续体当前状态",
            "",
            "当前本机已有五个波长的响应数组和对比图件。若外部 `DM-FEM2D` 大文件不完整，脚本会复用已有响应和历史图件；若数据完整，脚本会自动重新计算。",
            "300 m 波长按照既有溯源结果采用水动力节点反序候选，即 `reverse_hydrodynamic_node_order = true`，这是水动力节点块与结构主节点排列的顺序约定修正，不是简单把横坐标反画。",
            f"300 m 偏差专项诊断见：`{REGULAR_300_DIAGNOSTIC_PATH}`",
            "",
            f"连续性浮体单独报告：`{REGULAR_REPORT_PATH}`",
            "",
            "## 4. 铰接浮体约束验证",
            "",
            "铰接模型使用 `ExplicitHingeSpec` 定义节点对连接。每对铰接节点在全局刚度矩阵中加入 `+KC/-KC` 四个块，未释放自由度使用大刚度约束相对位移，释放转动自由度使用小惩罚刚度保留数值稳定性。",
            "",
            "### 4.1 约束设置",
            "",
            "| 算例 | 类型 | 模块数 | 铰接线 | 每线节点对 | 释放 DOF | 释放刚度 | 求解状态 |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for case_id, item in hinge_cases.items():
        manifest = item.get("manifest", {})
        hinges = manifest.get("hinges", []) if isinstance(manifest, dict) else []
        module_count = manifest.get("module_count", "") if isinstance(manifest, dict) else ""
        released_dofs = sorted({tuple(hinge.get("released_dofs_zero_based", [])) for hinge in hinges})
        released_text = ", ".join(str(list(value)) for value in released_dofs) if released_dofs else ""
        release_stiffness = hinges[0].get("released_dof_stiffness", "") if hinges else ""
        pair_counts = sorted({hinge.get("node_pair_count", "") for hinge in hinges})
        pair_text = ", ".join(str(value) for value in pair_counts)
        lines.append(
            f"| `{case_id}` | {hinge_case_type(case_id)} | {module_count} | "
            f"{len(hinges)} | {pair_text} | {released_text} | {release_stiffness} | {item.get('status')} |"
        )

    lines.extend(
        [
            "",
            "### 4.2 对比结果索引",
            "",
            "| 算例 | 对比对象 | 代表性图件 | 说明 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case_id, item in hinge_cases.items():
        figure = hinge_case_figure(case_id, item)
        lines.append(
            f"| `{case_id}` | {hinge_reference_summary(item)} | `{figure}` | {item.get('note', '')} |"
        )

    representative_figures = [
        ("单铰接中心线对比", hinge_case_figure("single_180", hinge_cases.get("single_180", {}))),
        ("双铰接 180 度中心线对比", hinge_case_figure("double_180", hinge_cases.get("double_180", {}))),
        ("双铰接 210 度中心线对比", hinge_case_figure("double_210", hinge_cases.get("double_210", {}))),
    ]
    lines.extend(["", "### 4.3 代表性图件", ""])
    for title, figure in representative_figures:
        if figure and Path(figure).exists():
            lines.extend([f"#### {title}", "", f"![{title}]({figure})", ""])

    lines.extend(
        [
            "### 4.4 铰接当前状态",
            "",
            "单铰、双铰和斜入射双铰均已完成标准脚本求解。双铰 180 度和 210 度有数字化他人数值曲线，双铰 180 度中心线还包含实验点。单铰接当前没有可靠的单铰数字化 CSV，因此报告采用当前 RODM 结果与历史论文图件的视觉对比，避免误用双铰数据。",
            "",
            f"铰接单独报告：`{HINGE_REPORT_PATH}`",
            "",
            "## 5. 统一结论",
            "",
            "- 连续性浮体 60 m、120 m、180 m、240 m、300 m 五个波长已经具备响应文件和图件型对比结果。",
            "- 铰接浮体已经实现单铰接和双铰接约束，并通过标准入口生成对比图件。",
            "- 当前代码已经把连续体水弹性计算、铰接约束装配、结果绘图和报告生成分开，后续可继续扩展到 10x10 模块和连接件刚度/位置优化。",
            "",
            "## 6. 复现命令",
            "",
            "```bash",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_regular_wave_batch_validation.py",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --case all",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/build_hydroelastic_validation_report.py",
            "```",
            "",
        ]
    )

    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8")
    DOC_REPORT_PATH.write_text(text, encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    """Build the report and copy it to docs for easy access."""

    regular_index = load_json(REGULAR_INDEX_PATH)
    hinge_metrics = load_json(HINGE_METRICS_PATH)
    if not regular_index:
        print(f"Missing regular-wave index: {REGULAR_INDEX_PATH}", file=sys.stderr)
        return 1
    if not hinge_metrics:
        print(f"Missing hinge metrics: {HINGE_METRICS_PATH}", file=sys.stderr)
        return 1

    report_path = write_report(regular_index, hinge_metrics)
    if REPORT_PATH != DOC_REPORT_PATH and REPORT_PATH.exists():
        shutil.copyfile(REPORT_PATH, DOC_REPORT_PATH)

    print(f"wrote_report={report_path}")
    print(f"wrote_docs_copy={DOC_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
