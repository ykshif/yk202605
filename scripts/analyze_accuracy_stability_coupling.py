"""Analyze accuracy-stability coupling for non-uniform SEREP-RODM layouts.

The non-uniform hydrodynamic module count is not a monotone refinement when
SEREP is used: each layout changes both the hydrodynamic control points and
the structural master DOFs.  This script combines actual heave RMSE results
with SEREP-ridge stability diagnostics so the paper can explain why some
larger-N non-uniform layouts degrade at the 300 m wavelength.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_minimum_feasible_nonuniform_optimization as feasible  # noqa: E402
import run_serep_nonuniform_wavelength_sweep as sweep  # noqa: E402
from offshore_energy_sim.reduction import reduce_matrix_dofs, separate_master_slave_dofs, transform_mass_matrix  # noqa: E402
from offshore_energy_sim.reduction.modal import _reorder_dof_blocks  # noqa: E402
from offshore_energy_sim.structure.matrix_io import read_abaqus_matrix_dense  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "accuracy_stability_coupling"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "accuracy_stability_coupling_report.md"

POOL_CSV = (
    REPO_ROOT
    / "results"
    / "minimum_feasible_nonuniform_optimization"
    / "tables"
    / "actual_candidate_pool_best_by_target_count.csv"
)

TOTAL_NODES = 793
FULL_DOFS_PER_NODE = 6
RETAINED_DOFS_PER_NODE = 5
REMOVED_FULL_DOFS_ZERO_BASED = (5,)
SEREP_RIDGE_RELATIVE_LAMBDA = 1.0e-16
TRANSFORMATION_FRO_LIMIT = 1.0e6
REDUCED_MASS_COND_LIMIT = 1.0e11


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def file_link(path: Path) -> str:
    return path.resolve().as_posix()


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(value)) for value in text.replace(",", " ").split() if value)


def load_retained_matrices() -> tuple[np.ndarray, np.ndarray]:
    paths = sweep.base.structural_paths()
    mass_full = read_abaqus_matrix_dense(paths.mass, dofs_per_node=FULL_DOFS_PER_NODE)
    stiffness_full = read_abaqus_matrix_dense(paths.stiffness, dofs_per_node=FULL_DOFS_PER_NODE)
    mass_retained = reduce_matrix_dofs(mass_full, TOTAL_NODES, REMOVED_FULL_DOFS_ZERO_BASED)
    stiffness_retained = reduce_matrix_dofs(stiffness_full, TOTAL_NODES, REMOVED_FULL_DOFS_ZERO_BASED)
    mass_retained = transform_mass_matrix(mass_retained, beta=0.0)
    return mass_retained, stiffness_retained


def ordered_serep_ridge_stats(
    *,
    mass_retained: np.ndarray,
    stiffness_retained: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
) -> dict[str, object]:
    from scipy.linalg import eigh

    reordered_stiffness, reordered_mass = _reorder_dof_blocks(
        stiffness_retained,
        mass_retained,
        np.asarray(master_dofs, dtype=int),
        np.sort(slave_dofs),
    )
    _eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
    for mode_index in range(eigenvectors.shape[1]):
        max_abs = np.max(np.abs(eigenvectors[:, mode_index]))
        if max_abs > 0.0:
            eigenvectors[:, mode_index] /= max_abs

    master_size = len(master_dofs)
    modes = eigenvectors[:, :master_size]
    master_modes = modes[:master_size, :]
    singular_values = np.linalg.svd(master_modes, compute_uv=False)
    normal_matrix = master_modes.T @ master_modes
    scale = np.linalg.norm(normal_matrix, ord=2)
    ridge = SEREP_RIDGE_RELATIVE_LAMBDA * scale * np.eye(master_size)
    mapping = np.linalg.solve(normal_matrix + ridge, master_modes.T)
    transformation = modes @ mapping
    reduced_mass = transformation.T @ reordered_mass @ transformation
    reduced_stiffness = transformation.T @ reordered_stiffness @ transformation
    return {
        "modal_block_smax": float(singular_values[0]),
        "modal_block_smin": float(singular_values[-1]),
        "modal_block_condition": float(singular_values[0] / singular_values[-1]),
        "transformation_fro_norm": float(np.linalg.norm(transformation, ord="fro")),
        "transformation_max_abs": float(np.max(np.abs(transformation))),
        "reduced_mass_condition": float(np.linalg.cond(reduced_mass)),
        "reduced_stiffness_condition": float(np.linalg.cond(reduced_stiffness)),
    }


def stability_stats(
    *,
    layout_id: str,
    module_count: int,
    master_nodes: tuple[int, ...],
    mass_retained: np.ndarray,
    stiffness_retained: np.ndarray,
) -> dict[str, object]:
    master_dofs, slave_dofs = separate_master_slave_dofs(
        TOTAL_NODES,
        master_nodes,
        dofs_per_node=RETAINED_DOFS_PER_NODE,
    )
    master_size = len(master_dofs)
    stats = ordered_serep_ridge_stats(
        mass_retained=mass_retained,
        stiffness_retained=stiffness_retained,
        master_dofs=master_dofs,
        slave_dofs=slave_dofs,
    )
    t_fro = float(stats["transformation_fro_norm"])
    mass_cond = float(stats["reduced_mass_condition"])
    stable = t_fro <= TRANSFORMATION_FRO_LIMIT and mass_cond <= REDUCED_MASS_COND_LIMIT
    return {
        "layout_id": layout_id,
        "module_count": module_count,
        "master_node_count": len(master_nodes),
        "master_dof_count": master_size,
        "selected_node_ids": " ".join(str(value) for value in master_nodes),
        "modal_block_smax": stats["modal_block_smax"],
        "modal_block_smin": stats["modal_block_smin"],
        "modal_block_condition": stats["modal_block_condition"],
        "transformation_fro_norm": t_fro,
        "transformation_max_abs": stats["transformation_max_abs"],
        "reduced_mass_condition": mass_cond,
        "reduced_stiffness_condition": stats["reduced_stiffness_condition"],
        "stable_by_threshold": stable,
        "transformation_fro_limit": TRANSFORMATION_FRO_LIMIT,
        "reduced_mass_cond_limit": REDUCED_MASS_COND_LIMIT,
    }


def unique_layout_rows(pool_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_layout: dict[str, dict[str, str]] = {}
    for row in pool_rows:
        layout_id = row["layout_id"]
        if layout_id not in by_layout and row.get("selected_node_ids"):
            by_layout[layout_id] = row
    return list(by_layout.values())


def compute_stability_table(pool_rows: list[dict[str, str]]) -> Path:
    mass_retained, stiffness_retained = load_retained_matrices()
    rows: list[dict[str, object]] = []
    for row in unique_layout_rows(pool_rows):
        rows.append(
            stability_stats(
                layout_id=row["layout_id"],
                module_count=int(row["module_count"]),
                master_nodes=parse_ints(row["selected_node_ids"]),
                mass_retained=mass_retained,
                stiffness_retained=stiffness_retained,
            )
        )
    return write_csv(TABLE_DIR / "serep_stability_by_layout.csv", rows)


def combine_accuracy_stability(pool_rows: list[dict[str, str]], stability_csv: Path) -> Path:
    stability_by_layout = {row["layout_id"]: row for row in read_csv(stability_csv)}
    u10 = stability_by_layout.get("uniform_U10")
    t_u10 = float(u10["transformation_fro_norm"]) if u10 is not None else float("nan")
    m_u10 = float(u10["reduced_mass_condition"]) if u10 is not None else float("nan")
    rows: list[dict[str, object]] = []
    for row in pool_rows:
        stability = stability_by_layout.get(row["layout_id"])
        if stability is None:
            continue
        t_fro = float(stability["transformation_fro_norm"])
        mass_cond = float(stability["reduced_mass_condition"])
        rows.append(
            {
                **row,
                "modal_block_condition": float(stability["modal_block_condition"]),
                "transformation_fro_norm": t_fro,
                "transformation_fro_ratio_vs_U10": t_fro / t_u10 if t_u10 > 0 else float("nan"),
                "transformation_max_abs": float(stability["transformation_max_abs"]),
                "reduced_mass_condition": mass_cond,
                "reduced_mass_condition_ratio_vs_U10": mass_cond / m_u10 if m_u10 > 0 else float("nan"),
                "reduced_stiffness_condition": float(stability["reduced_stiffness_condition"]),
                "stable_by_threshold": stability["stable_by_threshold"],
                "accuracy_gate_1p0": float(row["actual_rmse_ratio_vs_U10"]) <= 1.0,
                "accuracy_gate_0p95": float(row["actual_rmse_ratio_vs_U10"]) <= 0.95,
                "accuracy_gate_0p90": float(row["actual_rmse_ratio_vs_U10"]) <= 0.90,
            }
        )
    return write_csv(TABLE_DIR / "accuracy_stability_by_target_count.csv", rows)


def stable_minimum_rows(combined_csv: Path) -> list[dict[str, object]]:
    rows = read_csv(combined_csv)
    selected: list[dict[str, object]] = []
    for target_id in feasible.TARGETS:
        target_rows = [
            row
            for row in rows
            if row["target_id"] == target_id
            and row["layout_id"] != "uniform_U10"
            and row["stable_by_threshold"] == "True"
        ]
        for gate in feasible.RATIO_GATES:
            feasible_rows = [
                row
                for row in target_rows
                if float(row["actual_rmse_ratio_vs_U10"]) <= gate
            ]
            if feasible_rows:
                winner = min(
                    feasible_rows,
                    key=lambda row: (int(row["module_count"]), float(row["actual_rmse_ratio_vs_U10"])),
                )
                selected.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": gate,
                        "minimum_stable_nonuniform_module_count": winner["module_count"],
                        "selected_layout_id": winner["layout_id"],
                        "actual_rmse_ratio_vs_U10": winner["actual_rmse_ratio_vs_U10"],
                        "actual_improvement_vs_U10_percent": winner["actual_improvement_vs_U10_percent"],
                        "transformation_fro_norm": winner["transformation_fro_norm"],
                        "reduced_mass_condition": winner["reduced_mass_condition"],
                        "module_lengths_m": winner["module_lengths_m"],
                        "selected_node_ids": winner["selected_node_ids"],
                    }
                )
            else:
                selected.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": gate,
                        "minimum_stable_nonuniform_module_count": "",
                        "selected_layout_id": "",
                        "actual_rmse_ratio_vs_U10": "",
                        "actual_improvement_vs_U10_percent": "",
                        "transformation_fro_norm": "",
                        "reduced_mass_condition": "",
                        "module_lengths_m": "",
                        "selected_node_ids": "",
                    }
                )
    return selected


def value_by_target_count(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = np.full((len(feasible.TARGETS), len(feasible.COUNTS)), np.nan)
    for row in rows:
        target_id = row["target_id"]
        module_count = int(row["module_count"])
        if target_id not in feasible.TARGETS or module_count not in feasible.COUNTS:
            continue
        values[feasible.TARGETS.index(target_id), feasible.COUNTS.index(module_count)] = float(row[key])
    return values


def plot_accuracy_stability_heatmaps(combined_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(combined_csv)
    improvement = value_by_target_count(rows, "actual_improvement_vs_U10_percent")
    log_t = np.log10(value_by_target_count(rows, "transformation_fro_norm"))
    stable_mask = np.full_like(improvement, np.nan)
    for row in rows:
        target_id = row["target_id"]
        module_count = int(row["module_count"])
        if target_id not in feasible.TARGETS or module_count not in feasible.COUNTS:
            continue
        stable_mask[feasible.TARGETS.index(target_id), feasible.COUNTS.index(module_count)] = (
            1.0 if row["stable_by_threshold"] == "True" else 0.0
        )

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.2), sharey=True)
    vmax = min(100.0, max(10.0, float(np.nanmax(np.abs(improvement)))))
    image0 = axes[0].imshow(np.clip(improvement, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Heave RMSE improvement vs U10 (%)")
    image1 = axes[1].imshow(log_t, aspect="auto", cmap="magma")
    axes[1].set_title("SEREP stability: log10(||T||F)")
    for axis in axes:
        axis.set_xticks(np.arange(len(feasible.COUNTS)))
        axis.set_xticklabels([f"N{value}" for value in feasible.COUNTS])
        axis.set_yticks(np.arange(len(feasible.TARGETS)))
        axis.set_yticklabels(feasible.TARGETS)
        axis.set_xlabel("module/control-point count")
    axes[0].set_ylabel("target")

    for row_index in range(improvement.shape[0]):
        for col_index in range(improvement.shape[1]):
            if np.isfinite(improvement[row_index, col_index]):
                axes[0].text(col_index, row_index, f"{improvement[row_index, col_index]:.1f}", ha="center", va="center", fontsize=7)
            if np.isfinite(log_t[row_index, col_index]):
                suffix = "" if stable_mask[row_index, col_index] == 1.0 else " !"
                axes[1].text(col_index, row_index, f"{log_t[row_index, col_index]:.2f}{suffix}", ha="center", va="center", fontsize=7, color="white")
    cbar0 = fig.colorbar(image0, ax=axes[0])
    cbar0.set_label("positive means closer to U30")
    cbar1 = fig.colorbar(image1, ax=axes[1])
    cbar1.set_label("log10 Frobenius norm")
    fig.suptitle("Accuracy-stability map for non-uniform SEREP-RODM layouts", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURE_DIR / "accuracy_stability_heatmaps.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_accuracy_stability_scatter(combined_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = [row for row in read_csv(combined_csv) if row["layout_id"] != "uniform_U10"]
    fig, axis = plt.subplots(figsize=(9.0, 6.0))
    counts = sorted({int(row["module_count"]) for row in rows})
    cmap = plt.get_cmap("viridis", len(counts))
    count_to_color = {count: cmap(index) for index, count in enumerate(counts)}
    for target_id in feasible.TARGETS:
        target_rows = [row for row in rows if row["target_id"] == target_id]
        x_values = [np.log10(float(row["transformation_fro_norm"])) for row in target_rows]
        y_values = [float(row["actual_rmse_ratio_vs_U10"]) for row in target_rows]
        colors = [count_to_color[int(row["module_count"])] for row in target_rows]
        axis.scatter(x_values, y_values, s=38, color=colors, alpha=0.78, label=target_id)
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    axis.axhline(0.95, color="#777777", linestyle="--", linewidth=0.9)
    axis.axvline(np.log10(TRANSFORMATION_FRO_LIMIT), color="#d62728", linestyle="--", linewidth=1.0)
    axis.set_xlabel("log10(||T_SEREP||F)")
    axis.set_ylabel("actual heave RMSE ratio vs U10")
    axis.set_title("Accuracy-stability coupling")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = FIGURE_DIR / "accuracy_stability_scatter.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_stable_minimum_counts(stable_minimum_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(stable_minimum_csv)
    fig, axis = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(feasible.TARGETS))
    width = 0.24
    for offset, gate in zip((-width, 0.0, width), feasible.RATIO_GATES):
        y_values = []
        labels = []
        for target_id in feasible.TARGETS:
            row = next(
                item
                for item in rows
                if item["target_id"] == target_id and float(item["actual_ratio_gate"]) == gate
            )
            value = row["minimum_stable_nonuniform_module_count"]
            y_values.append(np.nan if value == "" else int(value))
            labels.append("not passed" if value == "" else f"N{value}")
        bars = axis.bar(x + offset, y_values, width=width, label=f"ratio <= {gate:g}")
        for bar, label in zip(bars, labels):
            if label == "not passed":
                continue
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, label, ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(feasible.TARGETS, rotation=18, ha="right")
    axis.set_ylabel("minimum stable non-uniform module count")
    axis.set_title("Minimum feasible N after SEREP stability screening")
    axis.set_ylim(0, max(feasible.COUNTS) + 1)
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "stable_minimum_counts_by_target.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    *,
    stability_csv: Path,
    combined_csv: Path,
    stable_minimum_csv: Path,
    figures: dict[str, Path],
) -> Path:
    combined_rows = read_csv(combined_csv)
    stability_rows = {row["layout_id"]: row for row in read_csv(stability_csv)}
    interesting = ["uniform_U10", "TNR_N11_01", "TNR_N12_09", "TNR_N13_18", "TNR_N14_26"]
    stability_table = []
    for layout_id in interesting:
        row = stability_rows.get(layout_id)
        if row is None:
            continue
        stability_table.append(
            (
                layout_id,
                row["master_node_count"],
                f"{float(row['transformation_fro_norm']):.3e}",
                f"{float(row['reduced_mass_condition']):.3e}",
                row["stable_by_threshold"],
            )
        )

    wl300_rows = [
        row
        for row in combined_rows
        if row["target_id"] == "wl_300m" and int(row["module_count"]) in {10, 11, 12, 13, 14}
    ]
    wl300_rows.sort(key=lambda row: int(row["module_count"]))
    wl300_table = [
        (
            f"N{row['module_count']}",
            row["layout_id"],
            f"{float(row['actual_rmse_ratio_vs_U10']):.4f}",
            f"{float(row['actual_improvement_vs_U10_percent']):.2f}%",
            f"{float(row['transformation_fro_norm']):.3e}",
            row["stable_by_threshold"],
        )
        for row in wl300_rows
    ]

    stable_rows = read_csv(stable_minimum_csv)
    min_table = []
    for target_id in feasible.TARGETS:
        row = next(
            item
            for item in stable_rows
            if item["target_id"] == target_id and float(item["actual_ratio_gate"]) == 0.95
        )
        min_table.append(
            (
                target_id,
                row["minimum_stable_nonuniform_module_count"] or "not passed",
                row["selected_layout_id"] or "-",
                (
                    "-"
                    if row["actual_rmse_ratio_vs_U10"] == ""
                    else f"{float(row['actual_rmse_ratio_vs_U10']):.4f}"
                ),
            )
        )

    lines = [
        "# 非均匀 SEREP-RODM 的精度-稳定性耦合分析",
        "",
        "## 1. 问题",
        "",
        (
            "N12/N13/N14 在 300 m 波长下并不构成单调加密序列。"
            "每个非均匀布局都会改变 SEREP 主控制节点集合，因此总误差同时包含水动力离散误差和结构降维重构误差。"
        ),
        "",
        "## 2. 关键图",
        "",
        f"![Accuracy-stability heatmaps]({file_link(figures['heatmaps'])})",
        "",
        f"![Accuracy-stability scatter]({file_link(figures['scatter'])})",
        "",
        f"![Stable minimum counts]({file_link(figures['stable_counts'])})",
        "",
        "## 3. 300 m 波长下的解释",
        "",
        markdown_table(
            ("case", "layout", "RMSE ratio", "improvement", "||T||F", "stable"),
            wl300_table,
        ),
        "",
        "N11 和 N12 的 SEREP 变换规模与 U10 同阶，因此它们在 300 m 波长下仍然优于 U10。N13/N14 的 SEREP 变换范数明显放大，结构重构误差抵消了水动力控制点增加的潜在收益。",
        "",
        "## 4. 代表性稳定性指标",
        "",
        markdown_table(
            ("layout", "master nodes", "||T||F", "cond(Mr)", "stable"),
            stability_table,
        ),
        "",
        "本文采用一个保守的稳定性筛选：",
        "",
        f"- `||T_SEREP||F <= {TRANSFORMATION_FRO_LIMIT:.1e}`",
        f"- `cond(M_reduced) <= {REDUCED_MASS_COND_LIMIT:.1e}`",
        "",
        "这两个阈值不是物理常数，而是用于剔除相对于 U10/N11/N12 突然放大的病态 SEREP 布局。",
        "",
        "## 5. 稳定性约束后的最小可行 N",
        "",
        markdown_table(("target", "minimum stable N for ratio<=0.95", "layout", "RMSE ratio"), min_table),
        "",
        "## 6. 论文表述建议",
        "",
        (
            "论文中不应把非均匀模块数量增加解释为必然单调收敛。更严谨的表述是："
            "非均匀控制点选择需要同时满足响应精度和 SEREP 降维稳定性约束；"
            "N13/N14 在 300 m 波长下的性能下滑正是稳定性约束必要性的证据。"
        ),
        "",
        "## 7. 输出文件",
        "",
        f"- SEREP 稳定性表：`{stability_csv}`",
        f"- 精度-稳定性合并表：`{combined_csv}`",
        f"- 稳定性约束后的最小 N：`{stable_minimum_csv}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return REPORT_PATH


def run_workflow(_args: argparse.Namespace) -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pool_rows = read_csv(POOL_CSV)
    stability_csv = compute_stability_table(pool_rows)
    combined_csv = combine_accuracy_stability(pool_rows, stability_csv)
    stable_rows = stable_minimum_rows(combined_csv)
    stable_minimum_csv = write_csv(TABLE_DIR / "stable_minimum_feasible_by_target_gate.csv", stable_rows)
    figures = {
        "heatmaps": plot_accuracy_stability_heatmaps(combined_csv),
        "scatter": plot_accuracy_stability_scatter(combined_csv),
        "stable_counts": plot_stable_minimum_counts(stable_minimum_csv),
    }
    report = write_report(
        stability_csv=stability_csv,
        combined_csv=combined_csv,
        stable_minimum_csv=stable_minimum_csv,
        figures=figures,
    )
    manifest = {
        "report": str(report),
        "stability_csv": str(stability_csv),
        "combined_csv": str(combined_csv),
        "stable_minimum_csv": str(stable_minimum_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (OUTPUT_ROOT / "accuracy_stability_coupling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> int:
    manifest = run_workflow(parse_args())
    for key, value in manifest.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
