"""Targeted actual RODM refinement for the weak non-uniform cases.

The minimum-feasible evidence package showed that two targets still need
stronger evidence:

* wl_300m
* band_equal_120_300m

This script performs a focused refinement in the plausible region N11..N14.
It keeps the module layout one-dimensional and FEM-aligned, and restricts the
module lengths to 20/30 m.  That makes the layouts easy to interpret: a fixed
number of short 20 m modules is redistributed along the 300 m body while all
module centers remain on the 5 m structural grid.

Candidate selection is still two-stage:

1. enumerate all 20/30 m layouts for each N and rank them with a U30 heave
   interpolation surrogate;
2. run true Capytaine + ordered SEREP-ridge RODM only for the selected small
   candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
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

import run_minimum_control_point_rodm_validation as mcpv  # noqa: E402
import run_serep_nonuniform_wavelength_sweep as sweep  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "targeted_nonuniform_refinement_wl300_band"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "targeted_nonuniform_refinement_report.md"

COUNTS = (11, 12, 13, 14)
WAVELENGTHS_M = (60, 120, 180, 240, 300)
TARGETS = ("wl_300m", "band_equal_120_300m")
TARGET_WAVELENGTHS = {
    "wl_300m": (300,),
    "band_equal_120_300m": (120, 180, 240, 300),
}
MODULE_LENGTHS = (20.0, 30.0)

PREVIOUS_MANIFESTS = (
    REPO_ROOT / "results" / "control_point_count_gate_study" / "tables" / "gate_layout_geometry_manifest.csv",
    REPO_ROOT
    / "results"
    / "control_point_count_gate_study_N13_only"
    / "tables"
    / "gate_layout_geometry_manifest.csv",
    REPO_ROOT
    / "results"
    / "minimum_feasible_nonuniform_optimization"
    / "tables"
    / "actual_candidate_pool_best_by_target_count.csv",
)


@dataclass(frozen=True)
class Candidate:
    lengths_m: tuple[float, ...]
    source_reasons: tuple[str, ...]
    surrogate_ratio_by_wavelength: dict[int, float]
    surrogate_score_by_target: dict[str, float]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def parse_lengths(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.replace(",", " ").split() if value)


def module_boundaries(lengths_m: tuple[float, ...]) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])


def module_centers(lengths_m: tuple[float, ...]) -> tuple[float, ...]:
    boundaries = module_boundaries(lengths_m)
    return tuple((0.5 * (boundaries[:-1] + boundaries[1:])).tolist())


def length_signature(lengths_m: tuple[float, ...]) -> str:
    return "_".join(str(int(value)) for value in lengths_m)


def enumerate_20_30_layouts(module_count: int) -> tuple[tuple[float, ...], ...]:
    """Enumerate 20/30 m layouts with total length exactly 300 m."""

    # 20*N + 10*n30 = 300.
    number_of_30m_modules = int(round((300.0 - 20.0 * module_count) / 10.0))
    if not 0 <= number_of_30m_modules <= module_count:
        return tuple()
    layouts: list[tuple[float, ...]] = []
    for positions in combinations(range(module_count), number_of_30m_modules):
        values = [20.0 for _ in range(module_count)]
        for position in positions:
            values[position] = 30.0
        layouts.append(tuple(values))
    return tuple(layouts)


def reference_curves() -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    u10_rmse_by_wavelength: dict[int, float] = {}
    for wavelength_m in WAVELENGTHS_M:
        x_ref, heave_ref = sweep.load_heave(mcpv.reference_response_path("U30_reference", wavelength_m))
        x_u10, heave_u10 = sweep.load_heave(mcpv.reference_response_path("uniform_U10", wavelength_m))
        if not np.allclose(x_ref, x_u10):
            raise ValueError(f"U10/U30 x mismatch at {wavelength_m} m")
        references[wavelength_m] = (x_ref, heave_ref)
        u10_rmse_by_wavelength[wavelength_m] = float(np.sqrt(np.mean((heave_u10 - heave_ref) ** 2)))
    return references, u10_rmse_by_wavelength


def interpolation_scores(
    lengths_m: tuple[float, ...],
    references: dict[int, tuple[np.ndarray, np.ndarray]],
    u10_rmse_by_wavelength: dict[int, float],
) -> tuple[dict[int, float], dict[str, float]]:
    centers = np.asarray(module_centers(lengths_m), dtype=float) / sweep.LENGTH_M
    ratio_by_wavelength: dict[int, float] = {}
    for wavelength_m, (x_ref, heave_ref) in references.items():
        sampled = np.interp(centers, x_ref, heave_ref)
        reconstructed = np.interp(x_ref, centers, sampled)
        rmse = float(np.sqrt(np.mean((reconstructed - heave_ref) ** 2)))
        ratio_by_wavelength[wavelength_m] = rmse / u10_rmse_by_wavelength[wavelength_m]
    score_by_target = {
        target_id: float(np.mean([ratio_by_wavelength[wavelength] for wavelength in wavelengths]))
        for target_id, wavelengths in TARGET_WAVELENGTHS.items()
    }
    return ratio_by_wavelength, score_by_target


def previous_hydro_by_lengths() -> dict[tuple[float, ...], Path]:
    by_lengths: dict[tuple[float, ...], Path] = {}
    for manifest in PREVIOUS_MANIFESTS:
        for row in read_csv(manifest):
            if not row.get("module_lengths_m") or not row.get("hydro_path"):
                continue
            path = Path(row["hydro_path"])
            if path.exists():
                by_lengths.setdefault(parse_lengths(row["module_lengths_m"]), path)
    return by_lengths


def previous_actual_best_layouts() -> dict[tuple[float, ...], list[str]]:
    reasons: dict[tuple[float, ...], list[str]] = {}
    pool = (
        REPO_ROOT
        / "results"
        / "minimum_feasible_nonuniform_optimization"
        / "tables"
        / "actual_candidate_pool_best_by_target_count.csv"
    )
    for row in read_csv(pool):
        if row.get("target_id") not in TARGETS:
            continue
        if int(row["module_count"]) not in COUNTS:
            continue
        lengths = parse_lengths(row.get("module_lengths_m", ""))
        if not lengths or set(lengths) - set(MODULE_LENGTHS):
            continue
        reasons.setdefault(lengths, []).append(f"previous_actual_{row['target_id']}_N{row['module_count']}")
    return reasons


def special_layouts() -> dict[tuple[float, ...], list[str]]:
    """Add interpretable layouts that the surrogate may not rank first."""

    layouts: dict[tuple[float, ...], list[str]] = {}
    for module_count in COUNTS:
        n30 = int(round((300.0 - 20.0 * module_count) / 10.0))
        if not 0 <= n30 <= module_count:
            continue

        def add(positions: tuple[int, ...], reason: str) -> None:
            values = [20.0 for _ in range(module_count)]
            for pos in positions:
                values[pos] = 30.0
            layouts.setdefault(tuple(values), []).append(reason)

        add(tuple(range(n30)), f"N{module_count}_30m_bow")
        add(tuple(range(module_count - n30, module_count)), f"N{module_count}_30m_stern")
        center_start = max(0, (module_count - n30) // 2)
        add(tuple(range(center_start, center_start + n30)), f"N{module_count}_30m_center")
        if n30 >= 2:
            edge_positions = tuple(range(n30 // 2)) + tuple(range(module_count - (n30 - n30 // 2), module_count))
            add(edge_positions, f"N{module_count}_30m_edges")
        if n30 > 0:
            distributed = tuple(np.round(np.linspace(0, module_count - 1, n30)).astype(int).tolist())
            add(distributed, f"N{module_count}_30m_distributed")
    return layouts


def build_candidate_pool(top_k_per_target_count: int) -> tuple[Candidate, ...]:
    references, u10_rmse = reference_curves()
    by_lengths: dict[tuple[float, ...], list[str]] = {}

    for lengths, reasons in previous_actual_best_layouts().items():
        by_lengths.setdefault(lengths, []).extend(reasons)
    for lengths, reasons in special_layouts().items():
        by_lengths.setdefault(lengths, []).extend(reasons)

    all_scored: dict[tuple[float, ...], tuple[dict[int, float], dict[str, float]]] = {}
    for module_count in COUNTS:
        layouts = enumerate_20_30_layouts(module_count)
        scored = []
        for lengths in layouts:
            ratio_by_wavelength, score_by_target = interpolation_scores(lengths, references, u10_rmse)
            all_scored[lengths] = (ratio_by_wavelength, score_by_target)
            scored.append((lengths, ratio_by_wavelength, score_by_target))
        for target_id in TARGETS:
            ranked = sorted(scored, key=lambda item: item[2][target_id])
            for rank, (lengths, _ratio, _score) in enumerate(ranked[:top_k_per_target_count], start=1):
                by_lengths.setdefault(lengths, []).append(f"surrogate_top_{target_id}_N{module_count}_r{rank}")

    candidates: list[Candidate] = []
    for lengths, reasons in sorted(by_lengths.items(), key=lambda item: (len(item[0]), item[0])):
        ratio_by_wavelength, score_by_target = all_scored.get(lengths, interpolation_scores(lengths, references, u10_rmse))
        candidates.append(
            Candidate(
                lengths_m=lengths,
                source_reasons=tuple(dict.fromkeys(reasons)),
                surrogate_ratio_by_wavelength=ratio_by_wavelength,
                surrogate_score_by_target=score_by_target,
            )
        )
    return tuple(candidates)


def layout_from_candidate(index: int, candidate: Candidate) -> sweep.LayoutSpec:
    module_count = len(candidate.lengths_m)
    return sweep.LayoutSpec(
        layout_id=f"TNR_N{module_count}_{index:02d}",
        display_name=f"targeted N{module_count}",
        category="targeted_refinement",
        module_lengths_m=candidate.lengths_m,
    )


def write_candidate_table(layouts: dict[str, sweep.LayoutSpec], candidates: dict[str, Candidate]) -> Path:
    rows: list[dict[str, object]] = []
    for layout_id, layout in layouts.items():
        candidate = candidates[layout_id]
        row = {
            "layout_id": layout_id,
            "module_count": layout.module_count,
            "source_reasons": " ; ".join(candidate.source_reasons),
            "module_lengths_m": " ".join(f"{value:g}" for value in candidate.lengths_m),
            "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(candidate.lengths_m)),
        }
        for target_id in TARGETS:
            row[f"surrogate_score_{target_id}"] = candidate.surrogate_score_by_target[target_id]
        for wavelength_m in WAVELENGTHS_M:
            row[f"surrogate_ratio_{wavelength_m}m"] = candidate.surrogate_ratio_by_wavelength[wavelength_m]
        rows.append(row)
    return write_csv(TABLE_DIR / "targeted_refinement_candidate_sources.csv", rows)


def write_geometry_manifest(
    layouts: tuple[sweep.LayoutSpec, ...],
    configs: dict[str, sweep.ArrayHydrodynamicsConfig],
    geometry_paths: dict[str, Path],
) -> Path:
    rows: list[dict[str, object]] = []
    for layout in layouts:
        geometry = sweep.geometry_rows(layout, configs[layout.layout_id])
        node_ids = [int(row["selected_node_id"]) for row in geometry]
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_count": layout.module_count,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(layout.module_lengths_m)),
                "selected_node_ids": " ".join(str(value) for value in node_ids),
                "max_abs_center_node_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
                "hydro_path": str(configs[layout.layout_id].output_path),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
            }
        )
    return write_csv(TABLE_DIR / "targeted_refinement_geometry_manifest.csv", rows)


def mean_rmse(metric_rows: list[dict[str, str]], layout_id: str, wavelengths_m: tuple[int, ...]) -> float:
    return float(np.mean([mcpv.metric_lookup(metric_rows, layout_id, value, "rmse_vs_U30") for value in wavelengths_m]))


def write_actual_tables(
    *,
    layouts: tuple[sweep.LayoutSpec, ...],
    metric_csv: Path,
) -> tuple[Path, Path]:
    metric_rows = mcpv.read_csv(metric_csv)
    actual_rows: list[dict[str, object]] = []
    for target_id, wavelengths in TARGET_WAVELENGTHS.items():
        u10_rmse = mean_rmse(metric_rows, "uniform_U10", wavelengths)
        for layout in layouts:
            if layout.layout_id == "U30_reference":
                continue
            actual_rmse = mean_rmse(metric_rows, layout.layout_id, wavelengths)
            ratio = actual_rmse / u10_rmse if u10_rmse > 0 else float("nan")
            actual_rows.append(
                {
                    "target_id": target_id,
                    "target_wavelengths_m": " ".join(str(value) for value in wavelengths),
                    "module_count": layout.module_count,
                    "layout_id": layout.layout_id,
                    "display_name": layout.display_name,
                    "actual_rmse_vs_U30": actual_rmse,
                    "uniform_U10_rmse_vs_U30": u10_rmse,
                    "actual_rmse_ratio_vs_U10": ratio,
                    "actual_improvement_vs_U10_percent": (1.0 - ratio) * 100.0,
                }
            )

    best_rows: list[dict[str, object]] = []
    for target_id in TARGETS:
        for module_count in COUNTS:
            matches = [
                row
                for row in actual_rows
                if row["target_id"] == target_id and int(row["module_count"]) == module_count
            ]
            if not matches:
                continue
            best = min(matches, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
            best_rows.append(dict(best))
    return (
        write_csv(TABLE_DIR / "targeted_refinement_actual_by_layout.csv", actual_rows),
        write_csv(TABLE_DIR / "targeted_refinement_best_by_target_count.csv", best_rows),
    )


def plot_heatmap(best_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(best_csv)
    values = np.full((len(TARGETS), len(COUNTS)), np.nan)
    labels = [["" for _ in COUNTS] for _ in TARGETS]
    for row in rows:
        target_index = TARGETS.index(row["target_id"])
        count_index = COUNTS.index(int(row["module_count"]))
        value = float(row["actual_improvement_vs_U10_percent"])
        values[target_index, count_index] = value
        labels[target_index][count_index] = f"{value:.1f}"
    finite = values[np.isfinite(values)]
    vmax = max(10.0, float(np.nanmax(np.abs(finite)))) if finite.size else 10.0
    vmax = min(vmax, 100.0)
    fig, axis = plt.subplots(figsize=(8.8, 3.6))
    image = axis.imshow(np.clip(values, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(COUNTS)))
    axis.set_xticklabels([f"N{value}" for value in COUNTS])
    axis.set_yticks(np.arange(len(TARGETS)))
    axis.set_yticklabels(TARGETS)
    axis.set_xlabel("module/control-point count")
    axis.set_title("Targeted refinement: best actual improvement vs U10 (%)")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if labels[row_index][col_index]:
                axis.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means closer to U30 than U10")
    fig.tight_layout()
    path = FIGURE_DIR / "targeted_refinement_best_improvement_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_heave_panel(
    *,
    response_paths: dict[tuple[str, int], Path],
    layouts: tuple[sweep.LayoutSpec, ...],
    best_csv: Path,
) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(best_csv)
    best_300 = min(
        [row for row in rows if row["target_id"] == "wl_300m"],
        key=lambda row: float(row["actual_rmse_ratio_vs_U10"]),
    )
    best_band = min(
        [row for row in rows if row["target_id"] == "band_equal_120_300m"],
        key=lambda row: float(row["actual_rmse_ratio_vs_U10"]),
    )
    plot_ids = ("U30_reference", "uniform_U10", best_300["layout_id"], best_band["layout_id"])
    layout_by_id = {layout.layout_id: layout for layout in layouts}
    colors = {
        "U30_reference": "#111111",
        "uniform_U10": "#1f77b4",
        best_300["layout_id"]: "#d62728",
        best_band["layout_id"]: "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
    for axis, wavelength_m in zip(np.ravel(axes), (120, 180, 240, 300)):
        used_labels = set()
        for layout_id in plot_ids:
            x, heave = sweep.load_heave(response_paths[(layout_id, wavelength_m)])
            if layout_id == "U30_reference":
                label = "U30 reference"
            else:
                label = layout_by_id[layout_id].display_name
            if label in used_labels:
                label = "_nolegend_"
            used_labels.add(label)
            axis.plot(
                x,
                heave,
                linewidth=2.0 if layout_id == "U30_reference" else 1.35,
                linestyle="-" if layout_id in {"U30_reference", "uniform_U10"} else "--",
                color=colors.get(layout_id, "#666666"),
                label=label,
            )
        axis.set_title(f"{wavelength_m} m")
        axis.set_ylabel("Heave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[1, 0].set_xlabel("x/L")
    axes[1, 1].set_xlabel("x/L")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Targeted non-uniform refinement: best heave curves", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURE_DIR / "targeted_refinement_best_heave_panel.png"
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
    candidate_csv: Path,
    geometry_csv: Path,
    actual_csv: Path,
    best_csv: Path,
    figures: dict[str, Path],
) -> Path:
    rows = mcpv.read_csv(best_csv)
    summary_rows = []
    for target_id in TARGETS:
        target_rows = [row for row in rows if row["target_id"] == target_id]
        best = min(target_rows, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
        summary_rows.append(
            (
                target_id,
                f"N{best['module_count']}",
                best["layout_id"],
                f"{float(best['actual_rmse_ratio_vs_U10']):.4f}",
                f"{float(best['actual_improvement_vs_U10_percent']):.2f}%",
            )
        )

    lines = [
        "# 针对 wl_300m 与 band_120_300m 的非均匀布局细化验证",
        "",
        "## 1. 目的",
        "",
        (
            "本轮只研究 N11-N14，并把模块长度限制为 20/30 m。"
            "这样可以检验：在不引入过复杂模块尺度的情况下，是否能进一步改善"
            "长波 `wl_300m` 和宽频带 `band_equal_120_300m` 的真实 RODM 误差。"
        ),
        "",
        "## 2. 图",
        "",
        f"![Targeted refinement heatmap]({file_link(figures['heatmap'])})",
        "",
        f"![Targeted refinement heave panel]({file_link(figures['heave_panel'])})",
        "",
        "## 3. 最优结果",
        "",
        markdown_table(("target", "best N", "best layout", "RMSE ratio vs U10", "improvement"), summary_rows),
        "",
        "## 4. 输出文件",
        "",
        f"- 候选来源：`{candidate_csv}`",
        f"- 几何与 FEM 控制点：`{geometry_csv}`",
        f"- 所有真实验证结果：`{actual_csv}`",
        f"- 每个 N 的最佳结果：`{best_csv}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    return REPORT_PATH


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    mcpv.configure_sweep_output(OUTPUT_ROOT)
    sweep.REPORT_PATH = REPORT_PATH
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_pool(args.top_k_per_target_count)
    layouts_by_id: dict[str, sweep.LayoutSpec] = {}
    candidates_by_id: dict[str, Candidate] = {}
    for index, candidate in enumerate(candidates, start=1):
        layout = layout_from_candidate(index, candidate)
        layouts_by_id[layout.layout_id] = layout
        candidates_by_id[layout.layout_id] = candidate

    candidate_csv = write_candidate_table(layouts_by_id, candidates_by_id)
    selected_layouts = tuple(layouts_by_id.values())
    layouts = (
        sweep.LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
        sweep.LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
        *selected_layouts,
    )
    mcpv.update_plot_colors(layouts)

    previous_hydro = previous_hydro_by_lengths()
    configs: dict[str, sweep.ArrayHydrodynamicsConfig] = {}
    for layout in layouts:
        config = sweep.hydro_config(layout, WAVELENGTHS_M, n_jobs=args.n_jobs)
        reusable = previous_hydro.get(layout.module_lengths_m)
        if reusable is not None and layout.layout_id not in {"U30_reference", "uniform_U10"}:
            config = replace(config, output_path=reusable)
        configs[layout.layout_id] = config

    geometry_by_layout = {layout.layout_id: sweep.geometry_rows(layout, configs[layout.layout_id]) for layout in layouts}
    geometry_paths = {
        layout.layout_id: sweep.write_geometry_csv(layout, geometry_by_layout[layout.layout_id]) for layout in layouts
    }
    geometry_csv = write_geometry_manifest(layouts, configs, geometry_paths)

    if args.dry_run:
        manifest = {
            "mode": "dry_run",
            "candidate_count": str(len(selected_layouts)),
            "candidate_csv": str(candidate_csv),
            "geometry_csv": str(geometry_csv),
        }
        (OUTPUT_ROOT / "targeted_refinement_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    response_paths = mcpv.copy_reference_responses(wavelengths_m=WAVELENGTHS_M)
    for layout in selected_layouts:
        config = configs[layout.layout_id]
        sweep.ensure_hydrodynamics(config, force=args.force_hydro)
        solved = sweep.solve_layout(
            layout,
            config,
            geometry_by_layout[layout.layout_id],
            WAVELENGTHS_M,
            force_response=args.force_response,
        )
        for wavelength_m, path in solved.items():
            response_paths[(layout.layout_id, wavelength_m)] = path

    metric_csv, layout_summary_csv = sweep.write_metrics(layouts, response_paths, WAVELENGTHS_M)
    actual_csv, best_csv = write_actual_tables(layouts=layouts, metric_csv=metric_csv)
    figures = {
        "heatmap": plot_heatmap(best_csv),
        "heave_panel": plot_heave_panel(
            response_paths=response_paths,
            layouts=layouts,
            best_csv=best_csv,
        ),
    }
    report = write_report(
        candidate_csv=candidate_csv,
        geometry_csv=geometry_csv,
        actual_csv=actual_csv,
        best_csv=best_csv,
        figures=figures,
    )
    manifest = {
        "mode": "full",
        "candidate_count": str(len(selected_layouts)),
        "report": str(report),
        "candidate_csv": str(candidate_csv),
        "geometry_csv": str(geometry_csv),
        "metric_csv": str(metric_csv),
        "layout_summary_csv": str(layout_summary_csv),
        "actual_csv": str(actual_csv),
        "best_csv": str(best_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (OUTPUT_ROOT / "targeted_refinement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k-per-target-count", type=int, default=2)
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
