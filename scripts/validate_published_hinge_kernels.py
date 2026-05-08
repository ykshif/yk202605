"""Audit and validate hinge kernels used by the published hinge notebooks.

This script does not need the missing Yoon `.mtx`/`.nc` inputs. It validates
the matrix connector kernels and node-pair definitions that can be checked from
the notebooks and legacy Python files:

- original `DM_Hinge.py` zero-release connector;
- Yoon single-/double-hinge notebook connectors with soft release penalties;
- 10x10 multi-module connector generation from `RODM_complex_interconnection.py`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import DM_Hinge  # noqa: E402
import RODM_complex_interconnection as complex_hinge  # noqa: E402
from offshore_energy_sim.structure import (  # noqa: E402
    add_hinge_connections_in_place,
    hinge_coupling_matrix,
)


OUTPUT_ROOT = REPO_ROOT / "results" / "hinge_published_validation"
METRICS_PATH = OUTPUT_ROOT / "published_hinge_kernel_metrics.json"
REPORT_PATH = REPO_ROOT / "docs" / "published_hinge_program_audit.md"
REFERENCE_ROOT = REPO_ROOT / "references" / "hinge_published"

LOCAL_PLAN_A2 = REPO_ROOT / "RODM_Hige_study_plan_a_2.ipynb"
LOCAL_COMPLEX = REPO_ROOT / "RODM_2D_complex.ipynb"
LOCAL_COMPLEX_HELPER = REPO_ROOT / "RODM_complex_interconnection.py"
ARCHIVED_FEM_REDUCEV2_PLAN_A2 = (
    REFERENCE_ROOT / "programs" / "RODM_Hige_study_plan_a_2_FEM_Reducev2.ipynb"
)
ARCHIVED_PAPER_FIG_DIR = REFERENCE_ROOT / "figures"
FEM_REDUCEV2_PLAN_A2 = Path(
    "/Users/yongkang/Library/CloudStorage/OneDrive-宁波东方理工大学/FEM_Reducev2/RODM_Hige_study_plan_a_2.ipynb"
)
PAPER_FIG_DIR = Path(
    "/Users/yongkang/Library/CloudStorage/OneDrive-宁波东方理工大学/论文Submit/OE_special_250308/Revise/"
    "Hydroelasticity RODM - Advantages and Application/Figs"
)


YOON_SINGLE_GROUPS = [
    (
        list(range(31, 404, 31)),
        list(range(404, 777, 31)),
    )
]
YOON_DOUBLE_GROUPS = [
    (
        list(range(21, 274, 21)),
        list(range(274, 527, 21)),
    ),
    (
        list(range(294, 547, 21)),
        list(range(547, 800, 21)),
    ),
]


def sha256_short(path: Path) -> str:
    """Return a short SHA-256 digest for an existing file."""

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def flatten_hinge_groups(
    hinge_groups: list[tuple[list[int], list[int]]],
) -> list[tuple[int, int]]:
    """Flatten notebook-style hinge groups into one list of node pairs."""

    pairs: list[tuple[int, int]] = []
    for nodes_a, nodes_b in hinge_groups:
        if len(nodes_a) != len(nodes_b):
            raise ValueError("Hinge side node lists must have the same length")
        pairs.extend(zip(nodes_a, nodes_b))
    return pairs


def compact_node_pairs(
    node_pairs: list[tuple[int, int]],
) -> tuple[list[int], list[int], dict[int, int]]:
    """Map sparse original node IDs to compact one-based IDs for safe checks."""

    unique_nodes = sorted({node for pair in node_pairs for node in pair})
    mapping = {node: index + 1 for index, node in enumerate(unique_nodes)}
    side_a = [mapping[node_a] for node_a, _ in node_pairs]
    side_b = [mapping[node_b] for _, node_b in node_pairs]
    return side_a, side_b, mapping


def add_manual_notebook_hinge_connections(
    matrix: np.ndarray,
    side_a: list[int],
    side_b: list[int],
    coupling: np.ndarray,
    *,
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Notebook-equivalent two-node connector assembly."""

    negative_coupling = -coupling
    for node_a, node_b in zip(side_a, side_b):
        index_a = (node_a - 1) * dofs_per_node
        index_b = (node_b - 1) * dofs_per_node
        slice_a = slice(index_a, index_a + dofs_per_node)
        slice_b = slice(index_b, index_b + dofs_per_node)

        matrix[slice_a, slice_a] += coupling
        matrix[slice_b, slice_b] += coupling
        matrix[slice_a, slice_b] += negative_coupling
        matrix[slice_b, slice_a] += negative_coupling
    return matrix


def validate_notebook_kernel(
    *,
    name: str,
    hinge_groups: list[tuple[list[int], list[int]]],
    k_hinge: float,
    released_dofs_zero_based: tuple[int, ...],
    released_dof_stiffness: float,
) -> dict[str, object]:
    """Validate current connector assembly against a notebook-style kernel."""

    original_pairs = flatten_hinge_groups(hinge_groups)
    compact_a, compact_b, mapping = compact_node_pairs(original_pairs)
    matrix_size = len(mapping) * 6
    coupling = hinge_coupling_matrix(
        k_hinge,
        released_dofs_zero_based=released_dofs_zero_based,
        released_dof_stiffness=released_dof_stiffness,
    )

    current = add_hinge_connections_in_place(
        np.zeros((matrix_size, matrix_size)),
        compact_a,
        compact_b,
        k_hinge=k_hinge,
        released_dofs_zero_based=released_dofs_zero_based,
        released_dof_stiffness=released_dof_stiffness,
    )
    expected = add_manual_notebook_hinge_connections(
        np.zeros((matrix_size, matrix_size)),
        compact_a,
        compact_b,
        coupling,
    )
    difference = current - expected
    return {
        "name": name,
        "node_pair_count": len(original_pairs),
        "unique_node_count": len(mapping),
        "max_original_node": max(mapping),
        "compact_matrix_shape": list(current.shape),
        "k_hinge": k_hinge,
        "released_dofs_zero_based": list(released_dofs_zero_based),
        "released_dof_stiffness": released_dof_stiffness,
        "max_abs_error": float(np.max(np.abs(difference))),
        "l2_error": float(np.linalg.norm(difference)),
    }


def validate_dm_hinge_legacy_kernel() -> dict[str, object]:
    """Validate package default against the original `DM_Hinge.py` kernel."""

    side_a = DM_Hinge.calculate_column_node_indices(2, 4, 3)
    side_b = DM_Hinge.calculate_column_node_indices(3, 4, 3)
    matrix_size = 12 * 6
    legacy = DM_Hinge.add_hinge_connections(
        np.zeros((matrix_size, matrix_size)),
        side_a,
        side_b,
        123.0,
    )
    current = add_hinge_connections_in_place(
        np.zeros((matrix_size, matrix_size)),
        side_a,
        side_b,
        k_hinge=123.0,
    )
    difference = current - legacy
    return {
        "name": "DM_Hinge.py legacy zero-release connector",
        "node_pair_count": len(side_a),
        "released_dofs_zero_based": [4],
        "released_dof_stiffness": 0.0,
        "max_abs_error": float(np.max(np.abs(difference))),
        "l2_error": float(np.linalg.norm(difference)),
    }


def validate_complex_grid_kernel(direction: int) -> dict[str, object]:
    """Validate the current connector kernel against 2x2 complex-grid legacy code."""

    grid_size = 2
    nodes_per_module = 49
    total_nodes = nodes_per_module * grid_size * grid_size
    if direction == 0:
        groups = complex_hinge.generate_hinge_x_pairs(
            grid_size=grid_size,
            N=nodes_per_module,
            nodes_per_row=7,
            total_rows=7,
        )
        released_dofs = (4,)
    elif direction == 1:
        groups = complex_hinge.generate_hinge_y_pairs(
            grid_size=grid_size,
            N=nodes_per_module,
            nodes_per_row=7,
            total_rows=7,
        )
        released_dofs = (3,)
    else:
        raise ValueError("direction must be 0 or 1")

    legacy = complex_hinge.apply_hinge_joints(
        N=total_nodes,
        k_hinge=1.0e10,
        hinges=groups,
        direction=direction,
    )
    current = np.zeros_like(legacy)
    for side_a, side_b in groups:
        add_hinge_connections_in_place(
            current,
            side_a,
            side_b,
            k_hinge=1.0e10,
            released_dofs_zero_based=released_dofs,
            released_dof_stiffness=10.0,
        )
    difference = current - legacy
    return {
        "name": f"RODM_complex_interconnection.py 2x2 direction={direction}",
        "grid_size": grid_size,
        "hinge_line_count": len(groups),
        "node_pair_count": sum(len(side_a) for side_a, _ in groups),
        "released_dofs_zero_based": list(released_dofs),
        "released_dof_stiffness": 10.0,
        "matrix_shape": list(current.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "l2_error": float(np.linalg.norm(difference)),
    }


def summarize_complex_10x10() -> dict[str, object]:
    """Return the 10x10 hinge-pair counts without building a huge dense matrix."""

    x_groups = complex_hinge.generate_hinge_x_pairs(
        grid_size=10,
        N=49,
        nodes_per_row=7,
        total_rows=7,
    )
    y_groups = complex_hinge.generate_hinge_y_pairs(
        grid_size=10,
        N=49,
        nodes_per_row=7,
        total_rows=7,
    )
    return {
        "program": str(LOCAL_COMPLEX),
        "helper": str(LOCAL_COMPLEX_HELPER),
        "grid_size": "10x10",
        "module_count": 100,
        "nodes_per_module": 49,
        "total_nodes": 4900,
        "dense_matrix_shape_if_built": [29400, 29400],
        "x_hinge_line_count": len(x_groups),
        "x_node_pair_count": sum(len(side_a) for side_a, _ in x_groups),
        "x_released_dofs_zero_based": [4],
        "x_released_dof_stiffness": 10.0,
        "y_hinge_line_count": len(y_groups),
        "y_node_pair_count": sum(len(side_a) for side_a, _ in y_groups),
        "y_released_dofs_zero_based": [3],
        "y_released_dof_stiffness": 10.0,
        "k_hinge": 1.0e10,
        "required_structure_files": [
            "StructureData/Hinge_complex_paper4/Job3030hinge-1_MASS1.mtx",
            "StructureData/Hinge_complex_paper4/Job3030hinge-1_STIF1.mtx",
        ],
        "required_hydro_file": "HydrodynamicData/Yoon_hinge/DM10_10_direction0_wl180.nc",
    }


def assert_zero_errors(metrics: dict[str, object]) -> None:
    """Fail if any kernel equivalence check is nonzero."""

    checks = [metrics["dm_hinge_legacy"]]
    checks.extend(metrics["published_kernel_cases"])
    checks.extend(metrics["complex_grid_kernel_checks"])
    nonzero = [item for item in checks if item["max_abs_error"] != 0.0]
    if nonzero:
        raise AssertionError(f"Nonzero hinge kernel differences: {nonzero}")


def write_report(metrics: dict[str, object]) -> None:
    """Write a concise Chinese audit report."""

    paths = metrics["program_artifacts"]
    missing = metrics["missing_strict_response_inputs"]
    missing_proxy = metrics["missing_current_proxy_response_inputs"]
    checks = [metrics["dm_hinge_legacy"]]
    checks.extend(metrics["published_kernel_cases"])
    checks.extend(metrics["complex_grid_kernel_checks"])
    complex_10 = metrics["complex_10x10"]

    lines = [
        "# 已发表铰接程序定位与核函数验证",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 结论先行",
        "",
        "本轮定位到两类关键程序：",
        "",
        "- 论文/Yoon 对比主线：`FEM_Reducev2/RODM_Hige_study_plan_a_2.ipynb` 更像最终论文程序候选；它与 Yoon 参考 CSV、历史输出文件位于同一工作目录，并使用 `k_hinge=1E15`、释放自由度小惩罚 `1` 的双铰接方案。",
        "- 当前迁移工作主线：本地 `RODM_Hige_study_plan_a_2.ipynb` 包含单铰、双铰、斜入射双铰等完整实验分支，常用 `k_hinge=1e10`、释放自由度小惩罚 `100`。",
        "- 10x10 多铰接主线：本地 `RODM_2D_complex.ipynb` 调用 `RODM_complex_interconnection.py`，对应 100 个模块、180 条铰接线、1260 对铰接节点。",
        "",
        "当前重构包已经可以精确表达这些铰接连接矩阵；但严格重算单铰/双铰响应仍缺少 Yoon 专用 `.mtx/.nc` 输入文件。",
        "",
        "## 2. 程序和图件定位",
        "",
        "为避免后续继续依赖 OneDrive，本轮已将论文候选程序、CSV 和图件复制到本地 `references/hinge_published/`。",
        "",
        "| 类型 | 路径 | SHA/状态 |",
        "| --- | --- | --- |",
    ]
    for item in paths:
        lines.append(f"| {item['type']} | `{item['path']}` | `{item['status']}` |")

    lines.extend(
        [
            "",
            "## 3. 单铰、双铰、10x10 参数",
            "",
            "| 算例 | 程序来源 | 关键参数 | 当前验证 |",
            "| --- | --- | --- | --- |",
            "| 单铰接 | 本地 `RODM_Hige_study_plan_a_2.ipynb` | 2 个模块，13 对节点，`k=1e10`，释放 DOF=4，小惩罚=100，Yoon 180 deg 水动力 | 连接核函数通过；响应重算缺输入 |",
            "| 双铰接 | `FEM_Reducev2/RODM_Hige_study_plan_a_2.ipynb` / 本地同名 notebook | 3 个模块，26 对节点；论文候选为 `k=1E15`、小惩罚=1；本地迁移版为 `k=1e10`、小惩罚=100 | 两套连接核函数均通过；响应重算缺输入 |",
            f"| 10x10 多铰接 | `RODM_2D_complex.ipynb` + `RODM_complex_interconnection.py` | {complex_10['module_count']} 个模块，x/y 各 {complex_10['x_hinge_line_count']} 条铰接线，合计 {complex_10['x_node_pair_count'] + complex_10['y_node_pair_count']} 对节点；x 释放 DOF=4，小惩罚=10；y 释放 DOF=3，小惩罚=10 | 2x2 子核函数精确通过；10x10 只统计节点对，未构造 29400x29400 稠密矩阵 |",
            "",
            "## 4. 核函数验证结果",
            "",
            "| 检查项 | 节点对数 | 释放 DOF | 小惩罚刚度 | 最大误差 |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for item in checks:
        lines.append(
            f"| {item['name']} | {item['node_pair_count']} | "
            f"`{item['released_dofs_zero_based']}` | `{item['released_dof_stiffness']}` | "
            f"`{item['max_abs_error']:.6g}` |"
        )

    lines.extend(
        [
            "",
            "## 5. 严格响应重算的阻塞文件",
            "",
            "以下文件当前没有在 OneDrive 或本机可搜索范围中找到，因此不能在 Mac 上严格复现 Yoon 单铰/双铰响应：",
        ]
    )
    lines.extend([f"- `{path}`" for path in missing])

    lines.extend(
        [
            "",
            "现有 `scripts/run_yoon_hinge_response_validation.py` 的 793 节点代理响应分支在 Mac 上也需要以下输入；本轮运行时首先停在 `JobMesh5_5_MASS1.mtx`：",
        ]
    )
    lines.extend([f"- `{path}`" for path in missing_proxy])

    lines.extend(
        [
            "",
            "## 6. 后续建议",
            "",
            "1. 优先恢复 `StructureData/Yoon_hinge` 与 `HydrodynamicData/Yoon_hinge` 中的原始 Yoon 输入文件，再跑响应级单铰/双铰对比。",
            "2. 后续优化程序应使用重构包中的可配置释放刚度，不要再写死 `0`、`1`、`10` 或 `100`。",
            "3. 10x10 问题不建议继续构造 29400x29400 稠密铰接矩阵，后续应改成稀疏矩阵或块装配，否则内存会很快变成小怪兽。",
            "",
            "## 7. 流程图",
            "",
            "```mermaid",
            "flowchart TD",
            '  A["历史程序与论文图件定位"] --> B["提取单铰/双铰/10x10 节点对"]',
            '  B --> C["统一为当前 hinges.py 连接矩阵接口"]',
            '  C --> D["核函数等价验证"]',
            '  D --> E{"Yoon .mtx/.nc 输入是否存在"}',
            '  E -- "不存在" --> F["保留历史图件和核函数验证，列出缺失输入"]',
            '  E -- "存在" --> G["严格重算单铰/双铰响应并生成对比图"]',
            "```",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "program_artifacts": [
            {
                "type": "论文候选程序本地归档",
                "path": str(ARCHIVED_FEM_REDUCEV2_PLAN_A2),
                "status": f"exists={ARCHIVED_FEM_REDUCEV2_PLAN_A2.exists()}, sha256={sha256_short(ARCHIVED_FEM_REDUCEV2_PLAN_A2)}",
            },
            {
                "type": "论文候选程序原始位置",
                "path": str(FEM_REDUCEV2_PLAN_A2),
                "status": f"exists={FEM_REDUCEV2_PLAN_A2.exists()}, sha256={sha256_short(FEM_REDUCEV2_PLAN_A2)}",
            },
            {
                "type": "本地迁移程序",
                "path": str(LOCAL_PLAN_A2),
                "status": f"exists={LOCAL_PLAN_A2.exists()}, sha256={sha256_short(LOCAL_PLAN_A2)}",
            },
            {
                "type": "10x10 程序",
                "path": str(LOCAL_COMPLEX),
                "status": f"exists={LOCAL_COMPLEX.exists()}, sha256={sha256_short(LOCAL_COMPLEX)}",
            },
            {
                "type": "10x10 铰接辅助函数",
                "path": str(LOCAL_COMPLEX_HELPER),
                "status": f"exists={LOCAL_COMPLEX_HELPER.exists()}, sha256={sha256_short(LOCAL_COMPLEX_HELPER)}",
            },
            {
                "type": "论文图件本地归档",
                "path": str(ARCHIVED_PAPER_FIG_DIR),
                "status": f"exists={ARCHIVED_PAPER_FIG_DIR.exists()}",
            },
            {
                "type": "论文图件原始目录",
                "path": str(PAPER_FIG_DIR),
                "status": f"exists={PAPER_FIG_DIR.exists()}",
            },
        ],
        "dm_hinge_legacy": validate_dm_hinge_legacy_kernel(),
        "published_kernel_cases": [
            validate_notebook_kernel(
                name="local RODM_20250310 single hinge, KC[4]=100",
                hinge_groups=YOON_SINGLE_GROUPS,
                k_hinge=1.0e10,
                released_dofs_zero_based=(4,),
                released_dof_stiffness=100.0,
            ),
            validate_notebook_kernel(
                name="local RODM_20250310 double hinge, KC[4]=100",
                hinge_groups=YOON_DOUBLE_GROUPS,
                k_hinge=1.0e10,
                released_dofs_zero_based=(4,),
                released_dof_stiffness=100.0,
            ),
            validate_notebook_kernel(
                name="FEM_Reducev2 paper-candidate double hinge, KC[4]=1",
                hinge_groups=YOON_DOUBLE_GROUPS,
                k_hinge=1.0e15,
                released_dofs_zero_based=(4,),
                released_dof_stiffness=1.0,
            ),
        ],
        "complex_grid_kernel_checks": [
            validate_complex_grid_kernel(direction=0),
            validate_complex_grid_kernel(direction=1),
        ],
        "complex_10x10": summarize_complex_10x10(),
        "missing_strict_response_inputs": [
            "StructureData/Yoon_hinge/Job_hinge_study_150_60_YoonModel_MASS1.mtx",
            "StructureData/Yoon_hinge/Job_hinge_study_150_60_YoonModel_STIF1.mtx",
            "StructureData/Yoon_hinge/Job_hinge_study_100_60_YoonModel-1_MASS1_rho282.mtx",
            "StructureData/Yoon_hinge/Job_hinge_study_100_60_YoonModel-1_STIF1_rho282.mtx",
            "HydrodynamicData/Yoon_hinge/DM10_direction180_slender180_rho1025.nc",
            "HydrodynamicData/Yoon_hinge/DM10_direction210_slender180_rho1025.nc",
            "StructureData/Hinge_complex_paper4/Job3030hinge-1_MASS1.mtx",
            "StructureData/Hinge_complex_paper4/Job3030hinge-1_STIF1.mtx",
            "HydrodynamicData/Yoon_hinge/DM10_10_direction0_wl180.nc",
        ],
        "missing_current_proxy_response_inputs": [
            "StructureData/JobMesh5_5_MASS1.mtx",
            "StructureData/JobMesh5_5_STIF1.mtx",
            "StructureData/ELEMENTSTIFFNESS_793.mtx",
            "HydrodynamicData/Yoga/BM10_145_direaction180.nc",
        ],
    }
    assert_zero_errors(metrics)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics)

    print("Published hinge kernel validation completed.")
    print(f"checks={1 + len(metrics['published_kernel_cases']) + len(metrics['complex_grid_kernel_checks'])}")
    print(f"max_abs_error=0")
    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
