"""Validate minimum-control-point layouts with full SEREP-ridge RODM solves.

This workflow connects the frequency-/mode-adaptive density indicator to the
actual hydroelastic solver.  It reads the dynamic-programming candidate layouts
from ``results/minimum_control_point_selection`` and checks whether the selected
layouts really reduce heave RMSE relative to uniform U10 when compared with the
ordered SEREP-ridge U30 reference solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_serep_nonuniform_wavelength_sweep as sweep  # noqa: E402


SELECTION_ROOT = REPO_ROOT / "results" / "minimum_control_point_selection"
SELECTION_CSV = SELECTION_ROOT / "tables" / "minimum_control_points_by_tolerance.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "minimum_control_point_rodm_validation"
REFERENCE_ROOT = REPO_ROOT / "results" / "uniform_reference_convergence_U5_U10_U15_U30_heave_serep_ridge_ordered"
PREVIOUS_SWEEP_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_wavelength_sweep"
DEFAULT_WAVELENGTHS_M = (60, 120, 180, 240, 300)


@dataclass(frozen=True)
class CandidateLayout:
    layout: sweep.LayoutSpec
    source_target_ids: tuple[str, ...]
    tolerance_ratio: float
    surrogate_cost_ratio: float


def file_uri(path: Path) -> str:
    return "file:///" + path.resolve().as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_wavelengths(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_WAVELENGTHS_M
    values = tuple(int(float(item)) for item in re.split(r"[\s,;]+", text.strip()) if item)
    if not values:
        raise ValueError("at least one wavelength is required")
    return values


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def parse_lengths(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.replace(",", " ").split() if value)


def module_centers(lengths_m: tuple[float, ...]) -> np.ndarray:
    boundaries = np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def configure_sweep_output(output_root: Path) -> None:
    sweep.OUTPUT_ROOT = output_root
    sweep.HYDRO_DIR = output_root / "hydro"
    sweep.RESPONSE_DIR = output_root / "responses"
    sweep.GEOMETRY_DIR = output_root / "geometry"
    sweep.FIGURE_DIR = output_root / "figures"
    sweep.TABLE_DIR = output_root / "tables"
    sweep.REPORT_PATH = output_root / "minimum_control_point_rodm_validation_report.md"
    for directory in (
        sweep.OUTPUT_ROOT,
        sweep.HYDRO_DIR,
        sweep.RESPONSE_DIR,
        sweep.GEOMETRY_DIR,
        sweep.FIGURE_DIR,
        sweep.TABLE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_selected_candidates(
    *,
    tolerance: float,
    target_filter: set[str] | None,
) -> list[CandidateLayout]:
    rows = read_csv(SELECTION_CSV)
    selected = [
        row
        for row in rows
        if np.isclose(float(row["tolerance_ratio_vs_U10_uniform"]), tolerance)
        and (target_filter is None or row["target_id"] in target_filter)
    ]
    if not selected:
        raise ValueError(f"no selected rows found for tolerance={tolerance}")

    by_lengths: dict[tuple[float, ...], list[dict[str, str]]] = {}
    for row in selected:
        by_lengths.setdefault(parse_lengths(row["module_lengths_m"]), []).append(row)

    candidates: list[CandidateLayout] = []
    for index, (lengths, group) in enumerate(by_lengths.items(), start=1):
        source_ids = tuple(row["target_id"] for row in group)
        primary = group[0]
        display_targets = "+".join(source_ids)
        layout_id = f"MCP_{slug(source_ids[0])}_tol{str(tolerance).replace('.', 'p')}_N{len(lengths)}"
        if len(group) > 1:
            layout_id = f"MCP_group{index}_tol{str(tolerance).replace('.', 'p')}_N{len(lengths)}"
        layout = sweep.LayoutSpec(
            layout_id=layout_id,
            display_name=f"MCP {display_targets}",
            category="minimum_control_point",
            module_lengths_m=lengths,
        )
        candidates.append(
            CandidateLayout(
                layout=layout,
                source_target_ids=source_ids,
                tolerance_ratio=tolerance,
                surrogate_cost_ratio=float(primary["cost_ratio_vs_U10_uniform"]),
            )
        )
    return candidates


def reference_response_path(layout_id: str, wavelength_m: int) -> Path:
    if layout_id == "U30_reference":
        candidates = [
            REFERENCE_ROOT / "responses" / "U30" / f"uniform_U30_wavelength_{wavelength_m}m_response.npy",
            PREVIOUS_SWEEP_ROOT
            / "responses"
            / "U30_reference"
            / f"U30_reference_wavelength_{wavelength_m}m_response.npy",
        ]
    elif layout_id == "uniform_U10":
        candidates = [
            REFERENCE_ROOT / "responses" / "U10" / f"uniform_U10_wavelength_{wavelength_m}m_response.npy",
            PREVIOUS_SWEEP_ROOT
            / "responses"
            / "uniform_U10"
            / f"uniform_U10_wavelength_{wavelength_m}m_response.npy",
        ]
    else:
        raise ValueError(layout_id)

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing reference response for {layout_id} at {wavelength_m} m")


def copy_reference_responses(
    *,
    wavelengths_m: tuple[int, ...],
) -> dict[tuple[str, int], Path]:
    response_paths: dict[tuple[str, int], Path] = {}
    for layout_id, source_case in (("U30_reference", "U30_reference"), ("uniform_U10", "uniform_U10")):
        for wavelength_m in wavelengths_m:
            source = reference_response_path(layout_id, wavelength_m)
            target = (
                sweep.RESPONSE_DIR
                / source_case
                / f"{source_case}_wavelength_{wavelength_m}m_response.npy"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            response_paths[(layout_id, wavelength_m)] = target
    return response_paths


def write_candidate_source_table(candidates: list[CandidateLayout]) -> Path:
    rows = []
    for candidate in candidates:
        layout = candidate.layout
        centers = module_centers(layout.module_lengths_m)
        rows.append(
            {
                "layout_id": layout.layout_id,
                "display_name": layout.display_name,
                "source_target_ids": " ".join(candidate.source_target_ids),
                "tolerance_ratio": candidate.tolerance_ratio,
                "surrogate_cost_ratio_vs_U10_uniform": candidate.surrogate_cost_ratio,
                "module_count": layout.module_count,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "module_centers_m": " ".join(f"{value:.1f}" for value in centers),
            }
        )
    return write_csv(sweep.TABLE_DIR / "validated_candidate_sources.csv", rows)


def write_geometry_manifest(
    layouts: tuple[sweep.LayoutSpec, ...],
    geometry_paths: dict[str, Path],
    configs: dict[str, sweep.ArrayHydrodynamicsConfig],
) -> Path:
    rows = []
    for layout in layouts:
        geometry = sweep.geometry_rows(layout, configs[layout.layout_id])
        node_ids = [int(row["selected_node_id"]) for row in geometry]
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_count": layout.module_count,
                "total_length_m": sum(layout.module_lengths_m),
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "selected_node_ids": " ".join(str(value) for value in node_ids),
                "max_abs_center_node_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
                "hydro_path": str(configs[layout.layout_id].output_path),
            }
        )
    return write_csv(sweep.TABLE_DIR / "validated_layout_geometry_manifest.csv", rows)


def update_plot_colors(layouts: tuple[sweep.LayoutSpec, ...]) -> None:
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#bcbd22",
    )
    sweep.COLORS["uniform_U10"] = "#1f77b4"
    for index, layout in enumerate(layouts):
        if layout.layout_id in {"U30_reference", "uniform_U10"}:
            continue
        sweep.COLORS[layout.layout_id] = palette[(index + 1) % len(palette)]


def primary_target_wavelength(source_target_ids: str) -> int | None:
    for token in source_target_ids.split():
        match = re.fullmatch(r"wl_(\d+)m", token)
        if match:
            return int(match.group(1))
    return None


def metric_lookup(rows: list[dict[str, str]], layout_id: str, wavelength_m: int, key: str) -> float:
    for row in rows:
        if row["layout_id"] == layout_id and int(row["wavelength_m"]) == wavelength_m:
            return float(row[key])
    raise KeyError((layout_id, wavelength_m, key))


def write_surrogate_actual_consistency(
    *,
    candidate_source_csv: Path,
    layout_summary_csv: Path,
    metrics_csv: Path,
) -> Path:
    candidate_rows = read_csv(candidate_source_csv)
    summary_rows = {row["layout_id"]: row for row in read_csv(layout_summary_csv)}
    metric_rows = read_csv(metrics_csv)
    rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        layout_id = candidate["layout_id"]
        summary = summary_rows[layout_id]
        target_wavelength = primary_target_wavelength(candidate["source_target_ids"])
        target_improvement = (
            metric_lookup(metric_rows, layout_id, target_wavelength, "improvement_vs_U10_percent")
            if target_wavelength is not None
            else float("nan")
        )
        mean_improvement = float(summary["mean_improvement_vs_U10_percent"])
        if mean_improvement > 0.0 and (target_wavelength is None or target_improvement > 0.0):
            note = "passes actual RODM screening"
        elif mean_improvement > 0.0:
            note = "mean passes, primary target fails; needs target-specific RODM screening"
        else:
            note = "surrogate-only failure; needs RODM-in-the-loop screening"
        rows.append(
            {
                "layout_id": layout_id,
                "display_name": candidate["display_name"],
                "source_target_ids": candidate["source_target_ids"],
                "surrogate_cost_ratio_vs_U10_uniform": candidate[
                    "surrogate_cost_ratio_vs_U10_uniform"
                ],
                "mean_actual_improvement_vs_U10_percent": mean_improvement,
                "better_than_U10_count": summary["better_than_U10_count"],
                "primary_target_wavelength_m": "" if target_wavelength is None else target_wavelength,
                "primary_target_improvement_vs_U10_percent": target_improvement,
                "validation_note": note,
            }
        )
    return write_csv(sweep.TABLE_DIR / "surrogate_actual_consistency.csv", rows)


def plot_clipped_applicability_heatmap(
    metrics_csv: Path,
    layouts: tuple[sweep.LayoutSpec, ...],
    wavelengths_m: tuple[int, ...],
    *,
    clip_percent: float = 50.0,
) -> Path:
    import matplotlib.pyplot as plt

    path = sweep.FIGURE_DIR / "minimum_control_point_applicability_heatmap_clipped.png"
    rows = read_csv(metrics_csv)
    plot_layouts = [layout for layout in layouts if layout.layout_id not in {"U30_reference", "uniform_U10"}]
    values = np.asarray(
        [
            [
                metric_lookup(rows, layout.layout_id, wavelength_m, "improvement_vs_U10_percent")
                for wavelength_m in wavelengths_m
            ]
            for layout in plot_layouts
        ],
        dtype=float,
    )
    color_values = np.clip(values, -clip_percent, clip_percent)

    fig, axis = plt.subplots(figsize=(12.5, 4.9))
    image = axis.imshow(
        color_values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-clip_percent,
        vmax=clip_percent,
    )
    axis.set_xticks(np.arange(len(wavelengths_m)))
    axis.set_xticklabels([str(value) for value in wavelengths_m])
    axis.set_yticks(np.arange(len(plot_layouts)))
    axis.set_yticklabels([layout.display_name for layout in plot_layouts])
    axis.set_xlabel("wavelength (m)")
    axis.set_title("Minimum-control-point validation: RMSE improvement vs uniform U10 (%)")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label(f"color clipped to +/-{clip_percent:g}%")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_surrogate_actual_consistency(consistency_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(consistency_csv)
    x = np.asarray([float(row["surrogate_cost_ratio_vs_U10_uniform"]) for row in rows])
    y = np.asarray([float(row["mean_actual_improvement_vs_U10_percent"]) for row in rows])
    labels = [row["display_name"].replace("MCP ", "") for row in rows]
    colors = ["#d62728" if "failure" in row["validation_note"] else "#2ca02c" for row in rows]

    fig, axis = plt.subplots(figsize=(9.6, 5.6))
    axis.axhline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    axis.scatter(x, y, s=90, c=colors, edgecolor="#222222", linewidth=0.7, zorder=3)
    for x_value, y_value, label in zip(x, y, labels):
        axis.annotate(label, (x_value, y_value), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("surrogate cost ratio vs U10 uniform")
    axis.set_ylabel("mean actual RMSE improvement vs U10 (%)")
    axis.set_title("Surrogate density objective vs actual RODM improvement")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    fig.tight_layout()
    path = sweep.FIGURE_DIR / "surrogate_vs_actual_rodm_improvement.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    *,
    tolerance: float,
    wavelengths_m: tuple[int, ...],
    candidate_source_csv: Path,
    geometry_manifest_csv: Path,
    consistency_csv: Path,
    metrics_csv: Path,
    layout_summary_csv: Path,
    best_by_wavelength_csv: Path,
    figures: dict[str, Path],
) -> Path:
    metrics = read_csv(metrics_csv)
    layout_summary = read_csv(layout_summary_csv)
    best_rows = read_csv(best_by_wavelength_csv)

    summary_table_rows = []
    for row in layout_summary:
        if row["layout_id"] == "U30_reference":
            continue
        summary_table_rows.append(
            (
                row["display_name"],
                f"{float(row['mean_rmse_vs_U30']):.6g}",
                f"{float(row['mean_improvement_vs_U10_percent']):.2f}%",
                row["better_than_U10_count"],
                row["selected_node_ids"],
            )
        )

    best_table_rows = [
        (
            row["wavelength_m"],
            row["best_display_name"],
            f"{float(row['best_rmse_vs_U30']):.6g}",
            f"{float(row['uniform_U10_rmse_vs_U30']):.6g}",
            f"{float(row['best_improvement_vs_U10_percent']):.2f}%",
        )
        for row in best_rows
    ]

    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# 最少控制点候选布局的 RODM 验证",
        "",
        "## 1. 研究目的",
        "",
        (
            "本验证把 frequency-/mode-adaptive 密度指标筛选出的最少控制点候选布局，"
            "接入完整 Capytaine + ordered SEREP-ridge RODM 求解。这里不再只看代理误差，"
            "而是直接比较 centerline heave RAO 相对 U30 参考解的 RMSE、最大误差和 roughness。"
        ),
        "",
        f"- 容限：`cost ratio <= {tolerance:g} * U10 uniform surrogate cost`",
        f"- 波长：`{', '.join(str(value) for value in wavelengths_m)} m`",
        "- 结构尺寸：`300 m x 60 m x 2 m`",
        "- 水动力模块布局：严格一维 `N x 1`，每个模块全宽 `60 m`",
        "- 结构降维：ordered `SEREP-ridge`, `relative_lambda=1e-16`",
        "",
        "## 2. 真实 RODM 误差图",
        "",
        f"![Heave response curves]({file_uri(figures['heave'])})",
        "",
        f"![RMSE curves]({file_uri(figures['rmse'])})",
        "",
        f"![Applicability heatmap]({file_uri(figures['heatmap_clipped'])})",
        "",
        f"![Surrogate vs actual]({file_uri(figures['surrogate_actual'])})",
        "",
        "## 3. 布局总体表现",
        "",
        table(
            ("case", "mean RMSE vs U30", "mean improvement vs U10", "better count", "selected FEM node ids"),
            summary_table_rows,
        ),
        "",
        "## 4. 逐波长最优布局",
        "",
        table(
            ("wavelength (m)", "best case", "best RMSE", "U10 RMSE", "improvement"),
            best_table_rows,
        ),
        "",
        "## 5. 关键输出文件",
        "",
        f"- 候选来源表：`{candidate_source_csv}`",
        f"- 几何与控制点清单：`{geometry_manifest_csv}`",
        f"- 代理指标与真实结果一致性表：`{consistency_csv}`",
        f"- 逐波长误差表：`{metrics_csv}`",
        f"- 布局汇总表：`{layout_summary_csv}`",
        f"- 逐波长最优表：`{best_by_wavelength_csv}`",
        f"- 图片目录：`{sweep.FIGURE_DIR}`",
        "",
        "## 6. 如何解读",
        "",
        (
            "若某个最少控制点布局在目标波长附近比 U10 更接近 U30，但在非目标波长不一定更优，"
            "这说明该算法适合作为 target-condition-oriented discretization，而不是一个全海况无条件最优划分。"
        ),
        (
            "若代理 cost ratio 与真实 RMSE 改善不一致，则需要把下一步算法从纯密度积分推进为"
            "`density preselection + RODM-in-the-loop verification`。"
        ),
        (
            "本次结果中 `MCP wl_300m` 就是这个反例：它的代理 cost ratio 满足容限，"
            "但真实 heave RMSE 明显恶化，因此论文中应把密度指标定位为候选生成器，"
            "最终选择必须通过 RODM 验证门控。"
        ),
        "",
    ]

    report_path = sweep.OUTPUT_ROOT / "minimum_control_point_rodm_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def write_dry_run_report(
    *,
    tolerance: float,
    wavelengths_m: tuple[int, ...],
    candidate_source_csv: Path,
    geometry_manifest_csv: Path,
    geometry_paths: dict[str, Path],
) -> Path:
    lines = [
        "# 最少控制点 RODM 验证 dry-run",
        "",
        f"- 容限：`{tolerance:g}`",
        f"- 波长：`{', '.join(str(value) for value in wavelengths_m)} m`",
        f"- 候选来源表：`{candidate_source_csv}`",
        f"- 几何与控制点清单：`{geometry_manifest_csv}`",
        "",
        "## 已生成几何表",
        "",
    ]
    for layout_id, path in geometry_paths.items():
        lines.append(f"- `{layout_id}`: `{path}`")
    lines.extend(
        [
            "",
            "正式运行会生成对应 `.nc` 水动力文件、RODM response `.npy`、误差 CSV 和三张对比图。",
            "",
        ]
    )
    path = sweep.OUTPUT_ROOT / "minimum_control_point_rodm_validation_dry_run.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    output_root = args.output_root or DEFAULT_OUTPUT_ROOT
    configure_sweep_output(output_root)
    wavelengths_m = parse_wavelengths(args.wavelengths)
    target_filter = set(args.targets.replace(",", " ").split()) if args.targets else None

    candidates = load_selected_candidates(tolerance=args.tolerance, target_filter=target_filter)
    candidate_layouts = tuple(candidate.layout for candidate in candidates)
    layouts = (
        sweep.LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
        sweep.LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
        *candidate_layouts,
    )
    update_plot_colors(layouts)

    configs = {
        layout.layout_id: sweep.hydro_config(layout, wavelengths_m, n_jobs=args.n_jobs)
        for layout in layouts
    }
    geometry_by_layout = {
        layout.layout_id: sweep.geometry_rows(layout, configs[layout.layout_id])
        for layout in layouts
    }
    geometry_paths = {
        layout.layout_id: sweep.write_geometry_csv(layout, geometry_by_layout[layout.layout_id])
        for layout in layouts
    }
    candidate_source_csv = write_candidate_source_table(candidates)
    geometry_manifest_csv = write_geometry_manifest(layouts, geometry_paths, configs)

    if args.dry_run:
        report = write_dry_run_report(
            tolerance=args.tolerance,
            wavelengths_m=wavelengths_m,
            candidate_source_csv=candidate_source_csv,
            geometry_manifest_csv=geometry_manifest_csv,
            geometry_paths=geometry_paths,
        )
        manifest = {
            "mode": "dry_run",
            "report": str(report),
            "candidate_source_csv": str(candidate_source_csv),
            "geometry_manifest_csv": str(geometry_manifest_csv),
        }
        (sweep.OUTPUT_ROOT / "minimum_control_point_rodm_validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    response_paths = copy_reference_responses(wavelengths_m=wavelengths_m)
    for layout in candidate_layouts:
        config = configs[layout.layout_id]
        sweep.ensure_hydrodynamics(config, force=args.force_hydro)
        solved = sweep.solve_layout(
            layout,
            config,
            geometry_by_layout[layout.layout_id],
            wavelengths_m,
            force_response=args.force_response,
        )
        for wavelength_m, path in solved.items():
            response_paths[(layout.layout_id, wavelength_m)] = path

    metrics_csv, layout_summary_csv = sweep.write_metrics(layouts, response_paths, wavelengths_m)
    best_by_wavelength_csv = sweep.write_best_by_wavelength(metrics_csv, wavelengths_m)
    consistency_csv = write_surrogate_actual_consistency(
        candidate_source_csv=candidate_source_csv,
        layout_summary_csv=layout_summary_csv,
        metrics_csv=metrics_csv,
    )
    figures = {
        "heatmap": sweep.plot_applicability_heatmap(metrics_csv, layouts, wavelengths_m),
        "heatmap_clipped": plot_clipped_applicability_heatmap(metrics_csv, layouts, wavelengths_m),
        "rmse": sweep.plot_rmse_curves(metrics_csv, layouts, wavelengths_m),
        "heave": sweep.plot_heave_grid(response_paths, layouts, wavelengths_m),
        "surrogate_actual": plot_surrogate_actual_consistency(consistency_csv),
    }
    report = write_report(
        tolerance=args.tolerance,
        wavelengths_m=wavelengths_m,
        candidate_source_csv=candidate_source_csv,
        geometry_manifest_csv=geometry_manifest_csv,
        consistency_csv=consistency_csv,
        metrics_csv=metrics_csv,
        layout_summary_csv=layout_summary_csv,
        best_by_wavelength_csv=best_by_wavelength_csv,
        figures=figures,
    )
    manifest = {
        "mode": "full",
        "report": str(report),
        "candidate_source_csv": str(candidate_source_csv),
        "geometry_manifest_csv": str(geometry_manifest_csv),
        "consistency_csv": str(consistency_csv),
        "metrics_csv": str(metrics_csv),
        "layout_summary_csv": str(layout_summary_csv),
        "best_by_wavelength_csv": str(best_by_wavelength_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (sweep.OUTPUT_ROOT / "minimum_control_point_rodm_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=0.8)
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional target ids from minimum_control_points_by_tolerance.csv, separated by comma/space.",
    )
    parser.add_argument(
        "--wavelengths",
        default=None,
        help="Comma/space separated wavelengths in m. Default: 60 120 180 240 300.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=sweep.CAPYTAINE_N_JOBS)
    parser.add_argument("--force-hydro", action="store_true")
    parser.add_argument("--force-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    manifest = run_workflow(parse_args())
    for key, value in manifest.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
