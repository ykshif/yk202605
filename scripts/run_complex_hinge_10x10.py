"""Run or inspect the standardized 10x10 modular hinge hydroelastic case."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import summarize_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.structure import scan_abaqus_matrix_file  # noqa: E402
from offshore_energy_sim.validation import (  # noqa: E402
    build_complex_hinge_10x10_case,
    missing_complex_hinge_input_paths,
    plot_complex_hinge_result,
    solve_complex_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "complex_hinge_10x10"


def case_manifest(case) -> dict[str, object]:
    """Return a compact JSON-friendly manifest for reproducibility."""

    x_hinges = [hinge for hinge in case.hinges if hinge.name.startswith("x ")]
    y_hinges = [hinge for hinge in case.hinges if hinge.name.startswith("y ")]
    return {
        "case_id": case.case_id,
        "title": case.title,
        "source_programs": case.source_programs,
        "grid": {
            "modules_per_side": case.grid.modules_per_side,
            "module_count": case.grid.module_count,
            "module_size": case.grid.module_size,
            "nodes_per_module_side": case.grid.nodes_per_module_side,
            "nodes_per_module": case.grid.nodes_per_module,
            "total_nodes": case.grid.total_nodes,
        },
        "paths": {
            "mass_matrix": case.mass_matrix_path,
            "stiffness_matrix": case.stiffness_matrix_path,
            "hydrodynamic": case.hydrodynamic_path,
        },
        "reduction": {
            "method": "static_condensation",
            "retained_dofs_per_node": case.retained_dofs_per_node,
            "removed_full_dofs_zero_based": case.removed_full_dofs_zero_based,
            "mass_projection_ordering": case.mass_projection_ordering,
        },
        "hydrodynamics": {
            "hydrodynamic_nodes": case.hydrodynamic_nodes,
            "hydrostatic_divisor": case.hydrostatic_divisor,
            "force_ordering": case.force_ordering,
        },
        "master_nodes": {
            "count": len(case.master_nodes_one_based),
            "first": case.master_nodes_one_based[0],
            "last": case.master_nodes_one_based[-1],
        },
        "hinges": {
            "total_lines": len(case.hinges),
            "x_lines": len(x_hinges),
            "y_lines": len(y_hinges),
            "node_pairs_per_x_line": len(x_hinges[0].node_pairs_one_based) if x_hinges else 0,
            "node_pairs_per_y_line": len(y_hinges[0].node_pairs_one_based) if y_hinges else 0,
            "total_node_pairs": sum(len(hinge.node_pairs_one_based) for hinge in case.hinges),
            "first_x_pair": x_hinges[0].node_pairs_one_based[0] if x_hinges else None,
            "first_y_pair": y_hinges[0].node_pairs_one_based[0] if y_hinges else None,
            "x_released_dofs_zero_based": x_hinges[0].released_dofs_zero_based if x_hinges else (),
            "y_released_dofs_zero_based": y_hinges[0].released_dofs_zero_based if y_hinges else (),
            "released_dof_stiffness": case.hinges[0].released_dof_stiffness if case.hinges else None,
        },
    }


def write_report(output_root: Path, metrics: dict[str, object]) -> Path:
    """Write a Chinese run report."""

    case = metrics["case"]
    status = metrics["status"]
    report_path = output_root / "report.md"
    lines = [
        "# 10x10 模块铰接水弹性标准算例报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 当前状态",
        "",
        f"- 算例：`{case['case_id']}`",
        f"- 状态：`{status}`",
        f"- 说明：{metrics.get('note', '')}",
        "",
        "## 2. 旧程序约定",
        "",
        "- 来源程序：`RODM_2D_complex.ipynb`、`RODM_complex_interconnection.py`。",
        "- 模块：10x10，共 100 个 30 m x 30 m 模块。",
        "- 结构网格：每个模块 7x7 节点，整体 4900 个结构节点。",
        "- 主节点：每个模块中心节点，共 100 个 hydrodynamic master nodes。",
        "- 铰接：x 方向 90 条线、y 方向 90 条线，共 1260 对节点连接。",
        "- 降阶：先删除第 6 自由度，保留 5DOF，再做 Static Condensation。",
        "- 水动力：`DM10_10_direction0_wl180.nc`，波长 180 m，入射方向 0 度。",
        "",
        "## 3. 输入文件",
        "",
    ]
    for label, path in case["paths"].items():
        lines.append(f"- {label}: `{path}`")

    missing = metrics.get("missing_inputs", [])
    if missing:
        lines.extend(["", "## 4. 缺失数据", ""])
        lines.extend([f"- `{path}`" for path in missing])
        lines.append("")
        lines.append("结构矩阵到位后可直接重新运行本脚本生成 10x10 响应和图片。")

    figures = metrics.get("figures", [])
    if figures:
        lines.extend(["", "## 4. 输出结果", ""])
        lines.extend([f"- 图像：`{path}`" for path in figures])
        lines.append(f"- 响应：`{metrics.get('response_path')}`")
        lines.append(f"- 合并网格：`{metrics.get('heave_grid_merged_path')}`")

    lines.extend(
        [
            "",
            "## 5. 运行命令",
            "",
            "```bash",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_complex_hinge_10x10.py",
            "```",
            "",
            "如果只检查输入和算例结构，不求解：",
            "",
            "```bash",
            "/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_complex_hinge_10x10.py --skip-solve",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="DM-FEM2D data root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    parser.add_argument("--skip-solve", action="store_true", help="Only write manifest/input checks")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return a nonzero exit code when required inputs are missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    case = build_complex_hinge_10x10_case(args.data_root)
    missing = missing_complex_hinge_input_paths(case)
    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": case_manifest(case),
        "missing_inputs": missing,
        "hydrodynamic_summary": asdict(
            summarize_hydrodynamic_dataset(
                case.hydrodynamic_path,
                load_metadata=case.hydrodynamic_path.exists(),
            )
        ),
        "mass_matrix_summary": asdict(scan_abaqus_matrix_file(case.mass_matrix_path)),
        "stiffness_matrix_summary": asdict(scan_abaqus_matrix_file(case.stiffness_matrix_path)),
    }

    if missing:
        metrics.update(
            {
                "status": "missing_inputs",
                "note": "等待 10x10 单模块结构质量/刚度矩阵传输完成。",
            }
        )
    elif args.skip_solve:
        metrics.update(
            {
                "status": "ready_not_solved",
                "note": "输入完整，本次按 --skip-solve 只完成结构检查。",
            }
        )
    else:
        result = solve_complex_hinge_case(case)
        response_path = output_root / "response.npy"
        heave_grid_raw_path = output_root / "heave_grid_raw.npy"
        heave_grid_merged_path = output_root / "heave_grid_merged.npy"
        np.save(response_path, result.response)
        np.save(heave_grid_raw_path, result.heave_grid_raw)
        np.save(heave_grid_merged_path, result.heave_grid_merged)
        figures = plot_complex_hinge_result(result, output_root / "figures")
        metrics.update(
            {
                "status": "solved",
                "note": "已完成 10x10 模块铰接水弹性计算和图片输出。",
                "omega": result.omega,
                "response_path": response_path,
                "heave_grid_raw_path": heave_grid_raw_path,
                "heave_grid_merged_path": heave_grid_merged_path,
                "figures": figures,
            }
        )

    metrics_path = output_root / "metrics.json"
    write_metrics_json(metrics_path, metrics)
    report_path = write_report(output_root, metrics)
    print("10x10 complex hinge workflow completed.")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    if missing and args.fail_on_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
