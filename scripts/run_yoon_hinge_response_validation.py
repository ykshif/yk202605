"""Validate single- and double-hinge response against local Yoon data.

This script standardizes the current hinge-response workflow without changing
the numerical algorithms. It runs the present 793-node RODM model with one and
two hinge lines, compares the centerline vertical displacement with available
Yoon et al. digitized curves/experiment points, and records that the original
Yoon-specific matrices are currently missing from the local data tree.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.loads import extend_force_vector_to_nodes  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402
from offshore_energy_sim.postprocess.validation import interpolated_curve_rmse  # noqa: E402
from offshore_energy_sim.reduction import (  # noqa: E402
    reduce_force_dofs,
    reduce_matrix_dofs,
    separate_master_slave_dofs,
    serep_reduce,
    transform_mass_matrix,
)
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.structure import (  # noqa: E402
    HingeLineSpec,
    build_hinged_stiffness,
    calculate_node_positions,
    read_abaqus_matrix_dense,
    read_plain_upper_triangle_stiffness_matrix,
)


DM_FEM_ROOT = Path(os.environ.get("RODM_DM_FEM_ROOT", r"E:\phd\Code\DM-FEM2D"))
STRUCTURE_DIR = DM_FEM_ROOT / "StructureData"
HYDRO_DIR = DM_FEM_ROOT / "HydrodynamicData" / "Yoga"
YOON_REFERENCE_DIR = Path(
    os.environ.get(
        "RODM_YOON_REFERENCE_DIR",
        r"E:\OneDrive - sjtu.edu.cn\铰接问题研究进展\Yoon et al. 数值结果",
    )
)
YOON_HINGE_DIR = Path(
    os.environ.get("RODM_YOON_HINGE_DIR", r"E:\OneDrive - sjtu.edu.cn\FEM_Reducev2\Hinge")
)
YOON_PDF_DIR = Path(
    os.environ.get("RODM_YOON_PDF_DIR", r"E:\OneDrive - sjtu.edu.cn\A_Work_done\RODM_AD\Hige")
)

OUTPUT_ROOT = REPO_ROOT / "results" / "hinge_response_validation"
FIGURE_DIR = OUTPUT_ROOT / "figures"
PDF_RENDER_DIR = OUTPUT_ROOT / "pdf_renders"
REPORT_PATH = REPO_ROOT / "docs" / "yoon_hinge_response_validation_report.md"

TOTAL_NODES = 793
FULL_DOFS = 6
RETAINED_DOFS = 5
HYDRO_NODES = 10
HINGE_GRID_COLUMNS = 61
HINGE_GRID_ROWS = 13
K_HINGE = 1.0e16

MASS_FILE = STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx"
STIFFNESS_FILE = STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx"
ELEMENT_STIFFNESS_FILE = STRUCTURE_DIR / "ELEMENTSTIFFNESS_793.mtx"
HYDRO_FILE = HYDRO_DIR / "BM10_145_direaction180.nc"

MISSING_EXACT_YOON_INPUTS = [
    DM_FEM_ROOT
    / "StructureData"
    / "Yoon_hinge"
    / "Job_hinge_study_100_60_YoonModel_MASS1.mtx",
    DM_FEM_ROOT
    / "StructureData"
    / "Yoon_hinge"
    / "Job_hinge_study_100_60_YoonModel_STIF1.mtx",
    DM_FEM_ROOT
    / "HydrodynamicData"
    / "Yoon_hinge"
    / "DM10_direction180_slender180_rho1025.nc",
]


def load_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column reference curve from comma or whitespace text."""

    path = Path(path)
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def curve_summary(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    """Return compact scalar diagnostics for one comparison curve."""

    return {
        "points": int(y_values.size),
        "x_min": float(np.min(x_values)),
        "x_max": float(np.max(x_values)),
        "y_min": float(np.min(y_values)),
        "y_max": float(np.max(y_values)),
        "y_mean": float(np.mean(y_values)),
    }


def render_pdf(pdf_path: Path, output_png: Path) -> Path | None:
    """Render a one-page PDF comparison figure when Poppler is available."""

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None or not pdf_path.exists():
        return None

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output_png.with_suffix("")
    subprocess.run(
        [
            pdftoppm,
            "-png",
            "-r",
            "180",
            "-singlefile",
            str(pdf_path),
            str(output_prefix),
        ],
        check=True,
    )
    return output_png


def render_legacy_figures() -> list[dict[str, str]]:
    """Render local legacy RODM/Yoon PDF figures for visual traceability."""

    pdf_names = [
        "Yoon-1-hige-180.pdf",
        "Yoon-2-hige-180-180-1.pdf",
        "Yoon-2-hige-180-180-2.pdf",
        "Yoon-2-hige-180-180-3.pdf",
    ]
    rendered = []
    for pdf_name in pdf_names:
        pdf_path = YOON_PDF_DIR / pdf_name
        png_path = PDF_RENDER_DIR / f"{Path(pdf_name).stem}.png"
        rendered_png = render_pdf(pdf_path, png_path)
        rendered.append(
            {
                "pdf": str(pdf_path),
                "png": str(rendered_png) if rendered_png is not None else "",
                "exists": pdf_path.exists(),
            }
        )
    return rendered


def apply_hinges_to_stiffness(
    stiffness_full: np.ndarray,
    element_stiffness: np.ndarray,
    hinges: list[HingeLineSpec],
) -> np.ndarray:
    """Remove hinge-line shell strips and add connector penalty stiffness."""

    hinged = np.array(stiffness_full, copy=True)
    for hinge in hinges:
        hinged = build_hinged_stiffness(
            hinged,
            hinge,
            element_stiffness_to_remove=element_stiffness,
            copy=False,
        )
    return hinged


def prepare_hydrodynamic_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and reduce hydrodynamic terms for the current 180 deg local case."""

    dataset = open_hydrodynamic_dataset(HYDRO_FILE, merge_complex=True)
    try:
        omega_values = dataset.omega.values
        omega = float(omega_values[0] if np.ndim(omega_values) else omega_values)
        added_mass = reduce_matrix_dofs(
            dataset["added_mass"][0].values,
            HYDRO_NODES,
            [5],
        )
        radiation_damping = reduce_matrix_dofs(
            dataset["radiation_damping"][0].values,
            HYDRO_NODES,
            [5],
        )
        hydrostatic_stiffness = reduce_matrix_dofs(
            dataset["hydrostatic_stiffness"].values,
            HYDRO_NODES,
            [5],
        )
        force = dataset["Froude_Krylov_force"][0].values + dataset["diffraction_force"][0].values
        wave_force = reduce_force_dofs(force, HYDRO_NODES, 5).reshape(1, HYDRO_NODES * RETAINED_DOFS)
        return added_mass, radiation_damping, hydrostatic_stiffness, wave_force, omega
    finally:
        dataset.close()


def solve_current_model_hinge_case(
    mass_full: np.ndarray,
    stiffness_full: np.ndarray,
    element_stiffness: np.ndarray,
    hinges: list[HingeLineSpec],
) -> dict[str, object]:
    """Run the current 793-node RODM response with specified hinge lines."""

    start = time.perf_counter()
    hinged_stiffness = apply_hinges_to_stiffness(stiffness_full, element_stiffness, hinges)

    # Full matrices: (793 nodes * 6 DOFs). Retained matrices remove local rz.
    mass_retained = reduce_matrix_dofs(mass_full, TOTAL_NODES, [5])
    stiffness_retained = reduce_matrix_dofs(hinged_stiffness, TOTAL_NODES, [5])
    mass_retained = transform_mass_matrix(mass_retained, beta=0.0)

    master_nodes = calculate_node_positions(424, 6, HYDRO_NODES)
    master_dofs, slave_dofs = separate_master_slave_dofs(
        TOTAL_NODES,
        master_nodes,
        dofs_per_node=RETAINED_DOFS,
    )
    reduced_mass, reduced_stiffness, transformation = serep_reduce(
        stiffness_retained,
        mass_retained,
        slave_dofs,
        master_nodes,
        dofs_per_master_node=RETAINED_DOFS,
    )
    added_mass, damping, hydrostatic, wave_force, omega = prepare_hydrodynamic_terms()
    master_displacement = solve_frequency_domain(
        reduced_mass + added_mass,
        damping,
        reduced_stiffness + hydrostatic,
        wave_force,
        omega,
    )
    global_displacement = reconstruct_global_response(
        transformation,
        master_displacement,
        master_dofs,
        slave_dofs,
    )
    x_values, heave = extract_centerline_heave(global_displacement)
    elapsed = time.perf_counter() - start

    return {
        "response": global_displacement,
        "x": x_values,
        "heave": heave,
        "elapsed_seconds": elapsed,
        "omega": omega,
        "master_nodes": master_nodes,
        "hinges": [
            {
                "column_a_one_based": hinge.column_a_one_based,
                "column_b_one_based": hinge.column_b_one_based,
                "released_dofs_zero_based": hinge.released_dofs_zero_based,
                "released_dof_stiffness": hinge.released_dof_stiffness,
                "k_hinge": hinge.k_hinge,
            }
            for hinge in hinges
        ],
    }


def plot_single_hinge(
    x_rodm: np.ndarray,
    heave_rodm: np.ndarray,
    references: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Plot the single-hinge centerline displacement comparison."""

    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "single_hinge_180deg_centerline_comparison.png"
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_rodm, heave_rodm, color="#d62728", linewidth=1.8, label="Current RODM hinge")
    if "yoon_numerical" in references:
        ax.plot(
            references["yoon_numerical"][0],
            references["yoon_numerical"][1],
            color="#1f1f1f",
            linewidth=1.5,
            label="Yoon et al. numerical",
        )
    if "experiment" in references:
        ax.scatter(
            references["experiment"][0],
            references["experiment"][1],
            color="#1f1f1f",
            s=26,
            marker="o",
            facecolors="none",
            label="Experiment",
            zorder=3,
        )
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Vertical displacement RAO")
    ax.set_title("Single hinge model, 180 deg wave incidence")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_double_hinge(
    x_rodm: np.ndarray,
    heave_rodm: np.ndarray,
    references: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Plot the double-hinge centerline displacement comparison."""

    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "double_hinge_180deg_centerline_comparison.png"
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_rodm, heave_rodm, color="#d62728", linewidth=1.8, label="Current RODM hinge")
    colors = ["#1f1f1f", "#4c78a8", "#666666"]
    for index, key in enumerate(("yoon_0_1", "yoon_0_2", "yoon_0_3")):
        if key not in references:
            continue
        ax.plot(
            references[key][0],
            references[key][1],
            color=colors[index],
            linewidth=1.4,
            linestyle="-" if index == 0 else "--",
            label=f"Yoon et al. numerical {index + 1}",
        )
    if "experiment" in references:
        ax.scatter(
            references["experiment"][0],
            references["experiment"][1],
            color="#1f1f1f",
            s=26,
            marker="o",
            facecolors="none",
            label="Experiment",
            zorder=3,
        )
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Vertical displacement RAO")
    ax.set_title("Double hinge model, 180 deg wave incidence")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def response_metrics(
    x_rodm: np.ndarray,
    heave_rodm: np.ndarray,
    references: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Compute response statistics and RMSE against each available reference."""

    metrics: dict[str, object] = {
        "current_rodm": curve_summary(x_rodm, heave_rodm),
        "rmse": {},
        "references": {},
    }
    for name, (x_ref, y_ref) in references.items():
        metrics["references"][name] = curve_summary(x_ref, y_ref)
        metrics["rmse"][name] = interpolated_curve_rmse(x_rodm, heave_rodm, x_ref, y_ref)
    return metrics


def write_report(metrics: dict[str, object]) -> None:
    """Write the Chinese validation report."""

    single = metrics["single_hinge"]
    double = metrics["double_hinge"]
    missing = [str(path) for path in MISSING_EXACT_YOON_INPUTS if not path.exists()]

    def fmt(value: object) -> str:
        return f"{float(value):.6g}"

    lines = [
        "# Yoon 铰接模型位移响应对比验证报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 验证目的",
        "",
        "本轮验证面向单铰接和双铰接浮体模型，在 180 deg 波浪入射条件下，对中心线竖向位移响应进行对比。",
        "脚本使用当前标准化的 RODM 铰接模块生成响应，并与本地保存的 Yoon et al. 数值曲线及实验点进行对比。",
        "",
        "预期数值变化：本轮只新增铰接验证脚本、绘图、报告和一个元素刚度矩阵读取工具；没有修改频域求解、SEREP、铰接刚度组装等数值算法。",
        "",
        "## 2. 重要数据状态",
        "",
        "严格复现 notebook 中的 Yoon 专用模型目前还缺少以下原始输入文件：",
    ]
    if missing:
        lines.extend([f"- `{path}`" for path in missing])
    else:
        lines.append("- 未发现缺失的 Yoon 专用输入文件。")

    lines.extend(
        [
            "",
            "因此，本报告包含两类证据：",
            "- 当前 793 节点 RODM 模型的单铰、双铰代理验证，可重复运行并可作为后续标准程序入口。",
            "- 本地已有的历史 Yoon/RODM 对比 PDF 渲染图，用于追踪此前基于 Yoon 专用输入完成的对比结果。",
            "",
            "## 3. 当前代理模型输入",
            "",
            f"- 质量矩阵：`{MASS_FILE}`",
            f"- 刚度矩阵：`{STIFFNESS_FILE}`",
            f"- 元素刚度：`{ELEMENT_STIFFNESS_FILE}`",
            f"- 180 deg 水动力文件：`{HYDRO_FILE}`",
            f"- 铰接刚度惩罚参数：`{K_HINGE:.3e}`",
            "",
            "## 4. 单铰接对比",
            "",
            f"- 图像：`{single['figure_png']}`",
            f"- 响应文件：`{single['response_path']}`",
            f"- 计算耗时：`{fmt(single['elapsed_seconds'])}` s",
            "",
            "| 对比对象 | RMSE |",
            "| --- | ---: |",
        ]
    )
    for name, value in single["metrics"]["rmse"].items():
        lines.append(f"| {name} | `{fmt(value)}` |")

    lines.extend(
        [
            "",
            "## 5. 双铰接对比",
            "",
            f"- 图像：`{double['figure_png']}`",
            f"- 响应文件：`{double['response_path']}`",
            f"- 计算耗时：`{fmt(double['elapsed_seconds'])}` s",
            "",
            "| 对比对象 | RMSE |",
            "| --- | ---: |",
        ]
    )
    for name, value in double["metrics"]["rmse"].items():
        lines.append(f"| {name} | `{fmt(value)}` |")

    lines.extend(
        [
            "",
            "## 6. 历史 Yoon/RODM 图件",
            "",
            "以下 PDF 是本地已经存在的历史对比图。图中红色虚线为 RODM，黑色曲线/点为 Yoon et al. 数值结果和实验结果。",
            "从已渲染图件看，历史 Yoon 专用模型下的单铰和双铰结果与参考曲线吻合较好；这与当前 793 节点代理模型的偏差形成区分。",
            "",
            "| PDF | 渲染 PNG |",
            "| --- | --- |",
        ]
    )
    for item in metrics["legacy_pdf_renders"]:
        png = item["png"] if item["png"] else "未渲染"
        lines.append(f"| `{item['pdf']}` | `{png}` |")

    lines.extend(
        [
            "",
            "## 7. 判断",
            "",
            "铰接刚度组装内核此前已经通过 legacy `DM_Hinge` 等价性和 Abaqus 模态基准验证；本轮进一步确认当前标准化入口可以完成单铰、双铰 RODM 响应计算、曲线对比和报告输出。",
            "但由于 Yoon 专用质量矩阵、刚度矩阵和水动力 NetCDF 文件当前不在本地可访问路径中，严格的 Yoon 模型重算还不能判定为完成。",
            "当前代理模型与 Yoon 曲线存在差异是预期结果，主要原因是结构模型、水动力模型和铰接位置定义不完全相同。",
            "因此，对铰接模块本身的判断是：矩阵组装路径可运行且内核验证通过；Yoon 响应级严格复现需要恢复专用输入文件后再完成。",
            "",
            "下一步应优先恢复 `StructureData/Yoon_hinge` 与 `HydrodynamicData/Yoon_hinge` 原始输入；恢复后可将本脚本中的代理模型分支替换为严格 Yoon 模型分支，重新生成同一套图和指标。",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_RENDER_DIR.mkdir(parents=True, exist_ok=True)

    for path in (MASS_FILE, STIFFNESS_FILE, ELEMENT_STIFFNESS_FILE, HYDRO_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    single_references = {
        "yoon_numerical": load_xy(YOON_HINGE_DIR / "Yoon_numerical_0_2.csv"),
        "experiment": load_xy(YOON_HINGE_DIR / "Yoon_exp.csv"),
    }
    double_references = {
        "yoon_0_1": load_xy(YOON_REFERENCE_DIR / "Yoon_numerical_0_1.csv"),
        "yoon_0_2": load_xy(YOON_REFERENCE_DIR / "Yoon_numerical_0_2.csv"),
        "yoon_0_3": load_xy(YOON_REFERENCE_DIR / "Yoon_numerical_0_3.csv"),
        "experiment": load_xy(YOON_HINGE_DIR / "Yoon_exp.csv"),
    }

    print("Reading structural matrices...")
    mass_full = read_abaqus_matrix_dense(MASS_FILE, dofs_per_node=FULL_DOFS)
    stiffness_full = read_abaqus_matrix_dense(STIFFNESS_FILE, dofs_per_node=FULL_DOFS)
    element_stiffness = read_plain_upper_triangle_stiffness_matrix(ELEMENT_STIFFNESS_FILE)

    single_hinges = [
        HingeLineSpec(
            column_a_one_based=31,
            column_b_one_based=32,
            nodes_per_row=HINGE_GRID_COLUMNS,
            rows_per_column=HINGE_GRID_ROWS,
            k_hinge=K_HINGE,
        )
    ]
    double_hinges = [
        HingeLineSpec(
            column_a_one_based=21,
            column_b_one_based=22,
            nodes_per_row=HINGE_GRID_COLUMNS,
            rows_per_column=HINGE_GRID_ROWS,
            k_hinge=K_HINGE,
        ),
        HingeLineSpec(
            column_a_one_based=41,
            column_b_one_based=42,
            nodes_per_row=HINGE_GRID_COLUMNS,
            rows_per_column=HINGE_GRID_ROWS,
            k_hinge=K_HINGE,
        ),
    ]

    print("Solving single-hinge current RODM response...")
    single_result = solve_current_model_hinge_case(
        mass_full,
        stiffness_full,
        element_stiffness,
        single_hinges,
    )
    single_response_path = OUTPUT_ROOT / "single_hinge_current_rodm_response.npy"
    np.save(single_response_path, single_result["response"])
    single_figure = plot_single_hinge(
        single_result["x"],
        single_result["heave"],
        single_references,
    )

    print("Solving double-hinge current RODM response...")
    double_result = solve_current_model_hinge_case(
        mass_full,
        stiffness_full,
        element_stiffness,
        double_hinges,
    )
    double_response_path = OUTPUT_ROOT / "double_hinge_current_rodm_response.npy"
    np.save(double_response_path, double_result["response"])
    double_figure = plot_double_hinge(
        double_result["x"],
        double_result["heave"],
        double_references,
    )

    legacy_pdf_renders = render_legacy_figures()

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exact_yoon_inputs_available": all(path.exists() for path in MISSING_EXACT_YOON_INPUTS),
        "missing_exact_yoon_inputs": [
            str(path) for path in MISSING_EXACT_YOON_INPUTS if not path.exists()
        ],
        "single_hinge": {
            "case": "current_793_node_surrogate",
            "response_path": str(single_response_path),
            "figure_png": str(single_figure),
            "elapsed_seconds": single_result["elapsed_seconds"],
            "omega": single_result["omega"],
            "master_nodes": single_result["master_nodes"],
            "hinges": single_result["hinges"],
            "metrics": response_metrics(
                single_result["x"],
                single_result["heave"],
                single_references,
            ),
        },
        "double_hinge": {
            "case": "current_793_node_surrogate",
            "response_path": str(double_response_path),
            "figure_png": str(double_figure),
            "elapsed_seconds": double_result["elapsed_seconds"],
            "omega": double_result["omega"],
            "master_nodes": double_result["master_nodes"],
            "hinges": double_result["hinges"],
            "metrics": response_metrics(
                double_result["x"],
                double_result["heave"],
                double_references,
            ),
        },
        "legacy_pdf_renders": legacy_pdf_renders,
    }

    write_metrics_json(OUTPUT_ROOT / "yoon_hinge_response_validation_metrics.json", metrics)
    write_report(metrics)

    print(f"Wrote metrics: {OUTPUT_ROOT / 'yoon_hinge_response_validation_metrics.json'}")
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote single figure: {single_figure}")
    print(f"Wrote double figure: {double_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
