"""Run the cleaned Yoon single-/double-hinge validation workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.validation import (  # noqa: E402
    build_yoon_hinge_cases,
    missing_input_paths,
    plot_yoon_hinge_case,
    solve_yoon_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "yoon_hinge_standard"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "references" / "hinge_published"


def case_manifest(case) -> dict[str, object]:
    """Return a compact JSON-friendly case manifest."""

    return {
        "case_id": case.case_id,
        "title": case.title,
        "module_count": case.module_count,
        "module_shape": [case.module_rows, case.module_columns],
        "total_nodes": case.total_nodes,
        "mass_matrix_path": case.mass_matrix_path,
        "stiffness_matrix_path": case.stiffness_matrix_path,
        "hydrodynamic_path": case.hydrodynamic_path,
        "reduction_method": case.reduction_method,
        "hydrostatic_divisor": case.hydrostatic_divisor,
        "reverse_force_node_order": case.reverse_force_node_order,
        "master_nodes_one_based": case.master_nodes_one_based,
        "hinges": [
            {
                "name": hinge.name,
                "node_pair_count": len(hinge.node_pairs_one_based),
                "first_pair": hinge.node_pairs_one_based[0],
                "last_pair": hinge.node_pairs_one_based[-1],
                "k_hinge": hinge.k_hinge,
                "released_dofs_zero_based": hinge.released_dofs_zero_based,
                "released_dof_stiffness": hinge.released_dof_stiffness,
            }
            for hinge in case.hinges
        ],
        "comparison_lines": [
            {
                "label": line.label,
                "row_index_zero_based": line.row_index_zero_based,
                "reverse_model_x": line.reverse_model_x,
                "reference_curves": line.reference_curves,
                "experiment_curves": line.experiment_curves,
                "legacy_figure_paths": line.legacy_figure_paths,
            }
            for line in case.comparison_lines
        ],
    }


def render_legacy_figures(case, output_dir: Path) -> list[Path]:
    """Render legacy PDF figures with macOS Quick Look when available."""

    qlmanage = shutil.which("qlmanage")
    if qlmanage is None:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for line in case.comparison_lines:
        for pdf_path in line.legacy_figure_paths:
            if not pdf_path.exists():
                continue
            subprocess.run(
                [
                    qlmanage,
                    "-t",
                    "-s",
                    "1200",
                    "-o",
                    str(output_dir),
                    str(pdf_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            quicklook_path = output_dir / f"{pdf_path.name}.png"
            if not quicklook_path.exists():
                continue
            target_path = output_dir / f"{case.case_id}_{line.label}_legacy.png"
            quicklook_path.replace(target_path)
            rendered.append(target_path)
    return rendered


def compose_current_legacy_panels(
    current_figures: list[Path],
    legacy_figures: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Stack current RODM figures with their rendered legacy paper figures."""

    try:
        from PIL import Image
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_by_stem = {
        path.name.replace("_legacy.png", ""): path
        for path in legacy_figures
        if path.name.endswith("_legacy.png")
    }
    panels: list[Path] = []
    for current_path in current_figures:
        legacy_path = legacy_by_stem.get(current_path.stem)
        if legacy_path is None:
            continue

        current = Image.open(current_path).convert("RGBA")
        legacy = Image.open(legacy_path).convert("RGBA")
        if legacy.width != current.width:
            scale = current.width / legacy.width
            legacy = legacy.resize(
                (current.width, max(1, int(legacy.height * scale))),
                Image.Resampling.LANCZOS,
            )

        gap = 28
        canvas = Image.new(
            "RGBA",
            (current.width, current.height + legacy.height + gap),
            (255, 255, 255, 255),
        )
        canvas.paste(current, (0, 0))
        canvas.paste(legacy, (0, current.height + gap))
        output_path = output_dir / f"{current_path.stem}_current_vs_legacy.png"
        canvas.save(output_path)
        panels.append(output_path)
    return panels


def write_report(output_root: Path, metrics: dict[str, object]) -> Path:
    """Write a Chinese run report."""

    report_path = output_root / "report.md"
    lines = [
        "# Yoon 单铰/双铰标准接口运行报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 本轮状态",
        "",
        "本脚本是 `RODM_Hige_study_plan_a_2.ipynb` 的标准化入口：保留已验证的单铰、双铰节点和数值约定，将重复 notebook 单元整理为可复用模块。",
        "",
        "| 算例 | 状态 | 说明 |",
        "| --- | --- | --- |",
    ]
    for case_id, item in metrics["cases"].items():
        note = item.get("note", "")
        lines.append(f"| `{case_id}` | `{item['status']}` | {note} |")

    lines.extend(
        [
            "",
            "## 2. 输入数据检查",
            "",
        ]
    )
    for case_id, item in metrics["cases"].items():
        missing = item.get("missing_inputs", [])
        lines.append(f"### {case_id}")
        if missing:
            lines.extend([f"- 缺失：`{path}`" for path in missing])
        else:
            lines.append("- 输入完整。")
        figures = item.get("figures", [])
        if figures:
            lines.extend([f"- 图像：`{path}`" for path in figures])
        legacy_figures = item.get("legacy_figures", [])
        if legacy_figures:
            lines.extend([f"- 历史论文图件渲染：`{path}`" for path in legacy_figures])
        panels = item.get("comparison_panels", [])
        if panels:
            lines.extend([f"- 当前结果与历史图件拼接：`{path}`" for path in panels])
        response_path = item.get("response_path")
        if response_path:
            lines.append(f"- 响应文件：`{response_path}`")
        lines.append("")

    lines.extend(
        [
            "## 3. 后续使用",
            "",
            "数据集传输完成后，直接重新运行：",
            "",
            "```bash",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --case all",
            "```",
            "",
            "如数据放在其他目录，增加 `--data-root /path/to/DM-FEM2D`。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", help="Case id: all, single_180, double_180, double_210")
    parser.add_argument("--data-root", default=None, help="DM-FEM2D data root")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT), help="Local reference CSV root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return a nonzero exit code when case inputs are missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_yoon_hinge_cases(args.data_root, args.reference_root)
    selected_ids = list(cases) if args.case == "all" else [args.case]

    unknown = [case_id for case_id in selected_ids if case_id not in cases]
    if unknown:
        raise ValueError(f"Unknown case id(s): {unknown}. Available: {sorted(cases)}")

    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": args.data_root,
        "reference_root": Path(args.reference_root),
        "output_root": output_root,
        "cases": {},
    }
    has_missing = False

    for case_id in selected_ids:
        case = cases[case_id]
        case_root = output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        missing = missing_input_paths(case)
        item = {
            "manifest": case_manifest(case),
            "missing_inputs": missing,
        }
        if missing:
            has_missing = True
            item.update(
                {
                    "status": "missing_inputs",
                    "note": "等待结构矩阵/水动力数据传输完成后再计算响应。",
                }
            )
            metrics["cases"][case_id] = item
            continue

        result = solve_yoon_hinge_case(case)
        response_path = case_root / "response.npy"
        heave_grid_path = case_root / "heave_grid.npy"
        np.save(response_path, result.response)
        np.save(heave_grid_path, result.heave_grid)
        figures = plot_yoon_hinge_case(result, case_root / "figures")
        legacy_figures = render_legacy_figures(case, case_root / "legacy_figures")
        comparison_panels = compose_current_legacy_panels(
            figures,
            legacy_figures,
            case_root / "comparison_panels",
        )
        has_digitized_reference = any(
            line.reference_curves or line.experiment_curves for line in case.comparison_lines
        )
        item.update(
            {
                "status": "solved",
                "note": "已完成响应计算和对比图输出。"
                if has_digitized_reference
                else "已完成响应计算；未找到数字化参考曲线，已附历史论文图件渲染。",
                "omega": result.omega,
                "response_path": response_path,
                "heave_grid_path": heave_grid_path,
                "figures": figures,
                "legacy_figures": legacy_figures,
                "comparison_panels": comparison_panels,
            }
        )
        metrics["cases"][case_id] = item

    metrics_path = output_root / "metrics.json"
    write_metrics_json(metrics_path, metrics)
    report_path = write_report(output_root, metrics)

    print("Yoon hinge standard workflow completed.")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    if has_missing:
        print("Some cases are waiting for transferred datasets.")
        return 1 if args.fail_on_missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
