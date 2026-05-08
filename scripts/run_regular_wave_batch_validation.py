"""Run or assemble regular-wave hydroelastic validation figures.

The script prefers a full local recomputation when the external DM-FEM2D data
tree is available. If those large inputs have not been copied to this Mac yet,
it falls back to the existing response arrays and existing comparison figures in
``results/regular_wave_batch``. The report intentionally omits RMSE tables and
keeps this workflow focused on figure-based validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import (  # noqa: E402
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    extract_centerline_heave,
    load_xy,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


WAVELENGTHS = [60, 120, 180, 240, 300]
HYDRO_NODE_REVERSE_BY_WAVELENGTH = {300: True}
DEFAULT_DM_FEM_ROOT = Path.home() / "data" / "DM-FEM2D"
DM_FEM_ROOT = Path(os.environ.get("RODM_DM_FEM_ROOT", str(DEFAULT_DM_FEM_ROOT)))
HYDRO_DIR = DM_FEM_ROOT / "HydrodynamicData" / "Yoga"
STRUCTURE_DIR = DM_FEM_ROOT / "StructureData"
COMPARISON_DIR = DM_FEM_ROOT / "data" / "Experiment_300_60"
OUTPUT_ROOT = REPO_ROOT / "results" / "regular_wave_batch"
SUMMARY_FIGURE_DIR = OUTPUT_ROOT / "figures"
DOC_REPORT_PATH = REPO_ROOT / "docs" / "regular_wave_batch_validation_report.md"
RESULT_REPORT_PATH = OUTPUT_ROOT / "regular_wave_batch_validation_report.md"
FIGURE_INDEX_PATH = OUTPUT_ROOT / "figure_index.json"


@dataclass(frozen=True)
class CasePaths:
    """File paths for one regular-wave wavelength case."""

    wavelength_m: int
    hydro_file: Path
    mass_file: Path
    stiffness_file: Path
    exp_file: Path
    fu_file: Path
    response_file: Path
    reversed_response_file: Path
    comparison_png: Path
    comparison_pdf: Path
    selected_png: Path
    selected_pdf: Path
    rodm_only_png: Path
    rodm_only_pdf: Path

    @property
    def reverse_hydrodynamic_node_order(self) -> bool:
        return HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(self.wavelength_m, False)

    @property
    def selected_response_file(self) -> Path:
        if self.reverse_hydrodynamic_node_order:
            return self.reversed_response_file
        return self.response_file

    @property
    def solve_inputs(self) -> tuple[Path, ...]:
        return (self.hydro_file, self.mass_file, self.stiffness_file)

    @property
    def comparison_inputs(self) -> tuple[Path, ...]:
        return (self.exp_file, self.fu_file)


def case_paths(wavelength_m: int) -> CasePaths:
    """Build all local and external paths for one wavelength."""

    case_root = OUTPUT_ROOT / f"wavelength_{wavelength_m}m"
    figure_dir = case_root / "figures"
    return CasePaths(
        wavelength_m=wavelength_m,
        hydro_file=HYDRO_DIR / f"DM10_{wavelength_m}_direction0.nc",
        mass_file=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        stiffness_file=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
        exp_file=COMPARISON_DIR / f"exp_{wavelength_m}.txt",
        fu_file=COMPARISON_DIR / f"fu_sim{wavelength_m}.txt",
        response_file=case_root / "response.npy",
        reversed_response_file=case_root / "variants" / "hydro_reversed" / "response.npy",
        comparison_png=figure_dir / f"regular_wave_{wavelength_m}m_heave_comparison.png",
        comparison_pdf=figure_dir / f"regular_wave_{wavelength_m}m_heave_comparison.pdf",
        selected_png=figure_dir / f"regular_wave_{wavelength_m}m_heave_selected.png",
        selected_pdf=figure_dir / f"regular_wave_{wavelength_m}m_heave_selected.pdf",
        rodm_only_png=figure_dir / f"regular_wave_{wavelength_m}m_heave_rodm_only.png",
        rodm_only_pdf=figure_dir / f"regular_wave_{wavelength_m}m_heave_rodm_only.pdf",
    )


def missing(paths: tuple[Path, ...]) -> list[Path]:
    """Return paths that do not exist."""

    return [path for path in paths if not path.exists()]


def build_case(paths: CasePaths) -> RodmFrequencyCase:
    """Create a configured RODM frequency-domain case."""

    return RodmFrequencyCase(
        case_id=f"regular_wave_{paths.wavelength_m}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=paths.hydro_file,
        structural_matrices=StructuralMatrixPaths(
            mass=paths.mass_file,
            stiffness=paths.stiffness_file,
        ),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        mass_blend_beta=0.0,
        use_hydrostatic=True,
        frequency_index=0,
        reverse_hydrodynamic_node_order=paths.reverse_hydrodynamic_node_order,
    )


def solve_case(paths: CasePaths) -> tuple[np.ndarray, float]:
    """Run one local hydroelastic calculation and save the response array."""

    start = time.perf_counter()
    result = solve_rodm_frequency_case(build_case(paths))
    elapsed = time.perf_counter() - start
    paths.selected_response_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths.selected_response_file, result.global_displacement)
    return result.global_displacement, elapsed


def load_or_solve_response(paths: CasePaths) -> tuple[np.ndarray | None, str, float | None]:
    """Prefer recomputation; fall back to existing response arrays."""

    solve_missing = missing(paths.solve_inputs)
    if not solve_missing:
        response, elapsed = solve_case(paths)
        return response, "重新计算", elapsed

    if paths.selected_response_file.exists():
        if paths.reverse_hydrodynamic_node_order:
            return np.load(paths.selected_response_file), "复用已有 hydro_reversed 响应", None
        return np.load(paths.selected_response_file), "复用已有响应", None

    if paths.response_file.exists():
        return np.load(paths.response_file), "缺少 hydro_reversed 响应，退回复用默认响应", None

    return None, "缺少外部输入且无已有响应", None


def plot_case(paths: CasePaths, response: np.ndarray, *, comparison_available: bool) -> tuple[Path, Path, str]:
    """Write one figure and return the selected figure path."""

    import matplotlib.pyplot as plt

    paths.comparison_png.parent.mkdir(parents=True, exist_ok=True)
    x_rodm, heave_rodm = extract_centerline_heave(response)

    if comparison_available:
        exp_x, exp_y = load_xy(paths.exp_file)
        fu_x, fu_y = load_xy(paths.fu_file)
        png_path = paths.selected_png if paths.reverse_hydrodynamic_node_order else paths.comparison_png
        pdf_path = paths.selected_pdf if paths.reverse_hydrodynamic_node_order else paths.comparison_pdf
        figure_kind = "重新绘制对比图"
    elif paths.comparison_png.exists() and not paths.reverse_hydrodynamic_node_order:
        return paths.comparison_png, paths.comparison_pdf, "沿用已有对比图"
    else:
        exp_x = exp_y = fu_x = fu_y = None
        png_path = paths.selected_png if paths.reverse_hydrodynamic_node_order else paths.rodm_only_png
        pdf_path = paths.selected_pdf if paths.reverse_hydrodynamic_node_order else paths.rodm_only_pdf
        figure_kind = "绘制300m方向修正图" if paths.reverse_hydrodynamic_node_order else "绘制RODM单曲线图"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        x_rodm,
        heave_rodm,
        color="#2ca02c" if paths.reverse_hydrodynamic_node_order else "#1f77b4",
        linewidth=1.8,
        label="RODM / hydro-node-reversed" if paths.reverse_hydrodynamic_node_order else "RODM / DM_Method",
    )
    if comparison_available:
        ax.scatter(
            exp_x,
            exp_y,
            color="#d62728",
            s=30,
            marker="o",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="Experiment",
        )
        ax.plot(
            fu_x,
            fu_y,
            color="#666666",
            linewidth=1.3,
            linestyle=":",
            label="Fu et al. simulation",
        )
    elif paths.reverse_hydrodynamic_node_order:
        baseline_path = REPO_ROOT / "displacement_55mesh_300.npy"
        if baseline_path.exists():
            x_baseline, heave_baseline = extract_centerline_heave(np.load(baseline_path))
            ax.plot(
                x_baseline,
                heave_baseline,
                color="#111111",
                linewidth=2.0,
                label="Saved baseline",
            )
        if paths.response_file.exists():
            x_default, heave_default = extract_centerline_heave(np.load(paths.response_file))
            ax.plot(
                x_default,
                heave_default,
                color="#d62728",
                linewidth=1.4,
                linestyle="--",
                label="Default node order",
            )

    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title(f"Regular wave validation, wavelength {paths.wavelength_m} m")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path, figure_kind


def build_summary_panel(items: list[dict[str, object]]) -> Path | None:
    """Combine the five wavelength figures into a single overview image."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    available = [item for item in items if Path(str(item["figure_png"])).exists()]
    if not available:
        return None

    SUMMARY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = SUMMARY_FIGURE_DIR / "regular_wave_60_300m_comparison_panel.png"
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.5))
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, available):
        image = mpimg.imread(str(item["figure_png"]))
        ax.imshow(image)
        ax.set_title(f"{item['wavelength_m']} m")
        ax.axis("off")

    for ax in axes_flat[len(available) :]:
        ax.axis("off")

    font_candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    title_font = FontProperties(fname=str(font_path)) if font_path is not None else None
    fig.suptitle(
        "连续性浮体规则波水弹性响应对比图汇总",
        fontsize=16,
        fontproperties=title_font,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(panel_path, dpi=220)
    plt.close(fig)
    return panel_path


def write_json(path: Path, data: object) -> None:
    """Write JSON with UTF-8 and stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(items: list[dict[str, object]], panel_path: Path | None) -> None:
    """Write a Chinese figure-only validation report."""

    lines = [
        "# 连续性浮体规则波水弹性对比验证图件",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 验证范围",
        "",
        "本报告整理 300 m x 60 m 连续性浮体在规则波波长 60 m、120 m、180 m、240 m、300 m 下的频域水弹性响应图件。",
        "报告按用户要求只保留图片结果和文件索引，不输出 RMSE 或其他误差指标。",
        "",
        "## 2. 本机运行策略",
        "",
        f"- 本机工作目录：`{REPO_ROOT}`",
        f"- 外部数据根目录：`{DM_FEM_ROOT}`",
        "- 若外部矩阵、水动力和对比曲线齐全，脚本会重新计算并绘制对比图。",
        "- 若外部大文件尚未迁移到 Mac，脚本会复用本机已有 `response.npy` 和历史对比图，避免覆盖已有验证图件。",
        "- 300 m 波长按既有溯源结论使用 `hydro_reversed` 响应；60-240 m 仍使用默认水动力节点顺序。",
        "",
    ]
    if panel_path is not None:
        lines.extend(["## 3. 汇总图", "", f"- `{panel_path}`", ""])

    lines.extend(
        [
            "## 4. 分波长图件",
            "",
            "| 波长 (m) | 响应来源 | 图件状态 | PNG 图件 |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for item in items:
        lines.append(
            f"| {item['wavelength_m']} | {item['response_status']} | "
            f"{item['figure_status']} | `{item['figure_png']}` |"
        )

    lines.extend(["", "## 5. 方向约定说明", ""])
    lines.append(
        "既有溯源报告显示，300 m 历史保存基准 `displacement_55mesh_300.npy` 更接近水动力节点反序候选结果；"
        "因此本图件流程将 300 m 标记为 `reverse_hydrodynamic_node_order = true`。"
    )
    lines.append(
        "这不是横坐标简单反画，而是 10 个水动力节点块与结构主节点排列之间的顺序约定差异。"
    )

    lines.extend(["", "## 6. 数据状态", ""])
    for item in items:
        missing_inputs = item["missing_inputs"]
        if missing_inputs:
            lines.append(f"波长 {item['wavelength_m']} m 缺少以下外部输入：")
            lines.extend(f"- `{path}`" for path in missing_inputs)
            lines.append("")
        else:
            lines.append(f"- 波长 {item['wavelength_m']} m：外部输入完整，已具备重新计算条件。")

    lines.extend(
        [
            "## 7. 结论",
            "",
            "本机副本中已经具备五个波长的响应文件和对比图件，可用于连续性浮体规则波水弹性结果查看。",
            "若需要完全重新求解，应先迁移 `DM-FEM2D` 外部数据目录并设置 `RODM_DM_FEM_ROOT`，再重新运行本脚本。",
            "",
        ]
    )

    text = "\n".join(lines)
    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT_PATH.write_text(text, encoding="utf-8")
    RESULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_REPORT_PATH.write_text(text, encoding="utf-8")


def process_case(wavelength_m: int) -> dict[str, object]:
    """Process one wavelength and return report metadata."""

    paths = case_paths(wavelength_m)
    response, response_status, elapsed = load_or_solve_response(paths)
    all_missing = missing(paths.solve_inputs + paths.comparison_inputs)
    comparison_available = not missing(paths.comparison_inputs)

    if response is None:
        figure_png = paths.comparison_png if paths.comparison_png.exists() else paths.rodm_only_png
        figure_pdf = paths.comparison_pdf if paths.comparison_pdf.exists() else paths.rodm_only_pdf
        figure_status = "缺少响应，未绘图"
    else:
        figure_png, figure_pdf, figure_status = plot_case(
            paths,
            response,
            comparison_available=comparison_available,
        )

    return {
        "wavelength_m": wavelength_m,
        "response_status": response_status,
        "elapsed_seconds": elapsed,
        "response_path": str(paths.selected_response_file),
        "figure_status": figure_status,
        "figure_png": str(figure_png),
        "figure_pdf": str(figure_pdf),
        "reverse_hydrodynamic_node_order": paths.reverse_hydrodynamic_node_order,
        "hydro_file": str(paths.hydro_file),
        "mass_file": str(paths.mass_file),
        "stiffness_file": str(paths.stiffness_file),
        "exp_file": str(paths.exp_file),
        "fu_file": str(paths.fu_file),
        "missing_inputs": [str(path) for path in all_missing],
    }


def main() -> int:
    """Run the figure-focused validation workflow."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = [process_case(wavelength_m) for wavelength_m in WAVELENGTHS]
    panel_path = build_summary_panel(items)

    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "dm_fem_root": str(DM_FEM_ROOT),
        "panel_png": str(panel_path) if panel_path is not None else "",
        "cases": items,
    }
    write_json(FIGURE_INDEX_PATH, index)
    write_report(items, panel_path)

    for item in items:
        print(
            f"{item['wavelength_m']} m: {item['response_status']}, "
            f"{item['figure_status']} -> {item['figure_png']}"
        )
    if panel_path is not None:
        print(f"summary_panel={panel_path}")
    print(f"figure_index={FIGURE_INDEX_PATH}")
    print(f"report={DOC_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
