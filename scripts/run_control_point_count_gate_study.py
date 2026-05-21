"""Run an actual RODM gate study for candidate control-point counts.

The density-based dynamic programming step proposes candidate layouts for each
module count.  This script verifies those candidates with the full workflow:

1. generate one Capytaine hydrodynamic NetCDF file for each distinct layout;
2. solve ordered SEREP-ridge RODM responses;
3. compare heave RMSE against the U30 reference;
4. report the smallest module count that passes each target-specific gate.

Changing the module count changes the hydrodynamic problem, so every distinct
candidate layout in this study gets its own hydrodynamic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import re
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_minimum_control_point_rodm_validation as mcpv  # noqa: E402
import run_serep_nonuniform_wavelength_sweep as sweep  # noqa: E402


CANDIDATE_CSV = (
    REPO_ROOT
    / "results"
    / "minimum_control_point_selection"
    / "tables"
    / "minimum_control_point_candidate_layouts.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "control_point_count_gate_study"
DEFAULT_TARGETS = (
    "wl_120m",
    "wl_180m",
    "wl_240m",
    "wl_300m",
    "band_equal_120_300m",
    "band_center_120_240m",
)
DEFAULT_COUNTS = (10, 11, 12, 14)
DEFAULT_ACTUAL_RATIO_GATES = (1.0, 0.95, 0.9)
TARGET_WAVELENGTHS = {
    "wl_120m": (120,),
    "wl_180m": (180,),
    "wl_240m": (240,),
    "wl_300m": (300,),
    "band_equal_120_300m": (120, 180, 240, 300),
    "band_center_120_240m": (120, 240),
}


@dataclass(frozen=True)
class GateCandidate:
    layout: sweep.LayoutSpec
    source_rows: tuple[dict[str, str], ...]


def parse_ints(text: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return default
    values = tuple(int(float(item)) for item in re.split(r"[\s,;]+", text.strip()) if item)
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_targets(text: str | None) -> tuple[str, ...]:
    if not text:
        return DEFAULT_TARGETS
    targets = tuple(item for item in re.split(r"[\s,;]+", text.strip()) if item)
    unknown = [target for target in targets if target not in TARGET_WAVELENGTHS]
    if unknown:
        raise ValueError(f"unknown target ids: {unknown}")
    return targets


def parse_floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    values = tuple(float(item) for item in re.split(r"[\s,;]+", text.strip()) if item)
    if not values:
        raise ValueError("at least one float is required")
    return values


def length_signature(lengths_m: tuple[float, ...]) -> str:
    return "_".join(str(int(value)) for value in lengths_m)


def is_uniform_u10(lengths_m: tuple[float, ...]) -> bool:
    return len(lengths_m) == 10 and all(np.isclose(value, 30.0) for value in lengths_m)


def load_gate_candidates(targets: tuple[str, ...], counts: tuple[int, ...]) -> tuple[GateCandidate, ...]:
    rows = [
        row
        for row in mcpv.read_csv(CANDIDATE_CSV)
        if row["target_id"] in targets and int(row["module_count"]) in counts
    ]
    by_lengths: dict[tuple[float, ...], list[dict[str, str]]] = {}
    for row in rows:
        lengths = mcpv.parse_lengths(row["module_lengths_m"])
        if is_uniform_u10(lengths):
            continue
        by_lengths.setdefault(lengths, []).append(row)

    candidates: list[GateCandidate] = []
    for index, (lengths, group) in enumerate(by_lengths.items(), start=1):
        module_count = len(lengths)
        source_ids = "+".join(row["target_id"] for row in group)
        layout_id = f"GATE_N{module_count}_{index:02d}"
        layout = sweep.LayoutSpec(
            layout_id=layout_id,
            display_name=f"N{module_count} {source_ids}",
            category="actual_gate_candidate",
            module_lengths_m=lengths,
        )
        candidates.append(GateCandidate(layout=layout, source_rows=tuple(group)))
    return tuple(candidates)


def write_candidate_source_table(candidates: tuple[GateCandidate, ...]) -> Path:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        layout = candidate.layout
        centers = mcpv.module_centers(layout.module_lengths_m)
        source_targets = " ".join(row["target_id"] for row in candidate.source_rows)
        source_counts = " ".join(row["module_count"] for row in candidate.source_rows)
        surrogate_ratios = " ".join(
            f"{float(row['cost_ratio_vs_U10_uniform']):.9g}" for row in candidate.source_rows
        )
        rows.append(
            {
                "layout_id": layout.layout_id,
                "display_name": layout.display_name,
                "module_count": layout.module_count,
                "source_target_ids": source_targets,
                "source_module_counts": source_counts,
                "source_surrogate_cost_ratios": surrogate_ratios,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "module_centers_m": " ".join(f"{value:.1f}" for value in centers),
            }
        )
    return mcpv.write_csv(sweep.TABLE_DIR / "gate_candidate_sources.csv", rows)


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
                "display_name": layout.display_name,
                "module_count": layout.module_count,
                "total_length_m": sum(layout.module_lengths_m),
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "selected_node_ids": " ".join(str(value) for value in node_ids),
                "max_abs_center_node_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
                "hydro_path": str(configs[layout.layout_id].output_path),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
            }
        )
    return mcpv.write_csv(sweep.TABLE_DIR / "gate_layout_geometry_manifest.csv", rows)


def target_wavelengths(target_id: str, available_wavelengths: tuple[int, ...]) -> tuple[int, ...]:
    wanted = TARGET_WAVELENGTHS[target_id]
    missing = [value for value in wanted if value not in available_wavelengths]
    if missing:
        raise ValueError(f"target {target_id} requires missing wavelengths {missing}")
    return wanted


def candidate_for_target_count(
    candidates: tuple[GateCandidate, ...],
    target_id: str,
    module_count: int,
) -> GateCandidate | None:
    matches = [
        candidate
        for candidate in candidates
        if candidate.layout.module_count == module_count
        and any(row["target_id"] == target_id for row in candidate.source_rows)
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous candidate for {target_id}, N{module_count}")
    return matches[0] if matches else None


def mean_rmse_for_layout(
    metric_rows: list[dict[str, str]],
    layout_id: str,
    wavelengths_m: tuple[int, ...],
) -> float:
    values = [
        mcpv.metric_lookup(metric_rows, layout_id, wavelength_m, "rmse_vs_U30")
        for wavelength_m in wavelengths_m
    ]
    return float(np.mean(values))


def write_actual_gate_tables(
    *,
    candidates: tuple[GateCandidate, ...],
    targets: tuple[str, ...],
    counts: tuple[int, ...],
    wavelengths_m: tuple[int, ...],
    metric_csv: Path,
    actual_ratio_gates: tuple[float, ...],
) -> tuple[Path, Path]:
    metric_rows = mcpv.read_csv(metric_csv)
    gate_rows: list[dict[str, object]] = []
    for target_id in targets:
        target_wls = target_wavelengths(target_id, wavelengths_m)
        u10_rmse = mean_rmse_for_layout(metric_rows, "uniform_U10", target_wls)
        for module_count in counts:
            candidate = candidate_for_target_count(candidates, target_id, module_count)
            if candidate is None:
                if module_count == 10:
                    layout_id = "uniform_U10"
                    display_name = "U10 uniform"
                    surrogate_ratio = 1.0
                else:
                    continue
            else:
                layout_id = candidate.layout.layout_id
                display_name = candidate.layout.display_name
                source_row = next(row for row in candidate.source_rows if row["target_id"] == target_id)
                surrogate_ratio = float(source_row["cost_ratio_vs_U10_uniform"])

            actual_rmse = mean_rmse_for_layout(metric_rows, layout_id, target_wls)
            actual_ratio = actual_rmse / u10_rmse if u10_rmse > 0.0 else float("nan")
            gate_rows.append(
                {
                    "target_id": target_id,
                    "target_wavelengths_m": " ".join(str(value) for value in target_wls),
                    "module_count": module_count,
                    "layout_id": layout_id,
                    "display_name": display_name,
                    "surrogate_cost_ratio_vs_U10_uniform": surrogate_ratio,
                    "actual_rmse_vs_U30": actual_rmse,
                    "uniform_U10_rmse_vs_U30": u10_rmse,
                    "actual_rmse_ratio_vs_U10": actual_ratio,
                    "actual_improvement_vs_U10_percent": (1.0 - actual_ratio) * 100.0,
                }
            )

    minimum_rows: list[dict[str, object]] = []
    for target_id in targets:
        target_rows = [row for row in gate_rows if row["target_id"] == target_id]
        for ratio_gate in actual_ratio_gates:
            feasible = [
                row
                for row in target_rows
                if float(row["actual_rmse_ratio_vs_U10"]) <= ratio_gate
            ]
            if feasible:
                selected = min(feasible, key=lambda row: (int(row["module_count"]), float(row["actual_rmse_ratio_vs_U10"])))
                minimum_rows.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_module_count": selected["module_count"],
                        "selected_layout_id": selected["layout_id"],
                        "selected_actual_rmse_ratio_vs_U10": selected["actual_rmse_ratio_vs_U10"],
                        "selected_actual_improvement_vs_U10_percent": selected[
                            "actual_improvement_vs_U10_percent"
                        ],
                    }
                )
            else:
                minimum_rows.append(
                    {
                        "target_id": target_id,
                        "actual_ratio_gate": ratio_gate,
                        "minimum_module_count": "",
                        "selected_layout_id": "",
                        "selected_actual_rmse_ratio_vs_U10": "",
                        "selected_actual_improvement_vs_U10_percent": "",
                    }
                )

    return (
        mcpv.write_csv(sweep.TABLE_DIR / "actual_gate_by_target_count.csv", gate_rows),
        mcpv.write_csv(sweep.TABLE_DIR / "actual_gate_minimum_counts.csv", minimum_rows),
    )


def plot_gate_heatmap(gate_csv: Path, targets: tuple[str, ...], counts: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(gate_csv)
    values = np.full((len(targets), len(counts)), np.nan)
    labels = [["" for _ in counts] for _ in targets]
    for row in rows:
        target_index = targets.index(row["target_id"])
        count_index = counts.index(int(row["module_count"]))
        value = float(row["actual_improvement_vs_U10_percent"])
        values[target_index, count_index] = value
        labels[target_index][count_index] = f"{value:.1f}"

    finite = values[np.isfinite(values)]
    vmax = max(10.0, float(np.nanmax(np.abs(finite)))) if finite.size else 10.0
    vmax = min(vmax, 80.0)
    color_values = np.clip(values, -vmax, vmax)

    fig, axis = plt.subplots(figsize=(9.8, 5.4))
    image = axis.imshow(color_values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(counts)))
    axis.set_xticklabels([f"N{count}" for count in counts])
    axis.set_yticks(np.arange(len(targets)))
    axis.set_yticklabels(targets)
    axis.set_xlabel("module/control-point count")
    axis.set_title("Actual RODM gate: heave RMSE improvement vs U10 (%)")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if labels[row_index][col_index]:
                axis.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means closer to U30 than U10")
    fig.tight_layout()
    path = sweep.FIGURE_DIR / "actual_gate_improvement_by_target_count.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_gate_curves(gate_csv: Path, targets: tuple[str, ...], counts: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(gate_csv)
    fig, axis = plt.subplots(figsize=(10.8, 5.8))
    for target_id in targets:
        target_rows = [row for row in rows if row["target_id"] == target_id]
        y_values = []
        x_values = []
        for count in counts:
            current = [row for row in target_rows if int(row["module_count"]) == count]
            if not current:
                continue
            x_values.append(count)
            y_values.append(float(current[0]["actual_rmse_ratio_vs_U10"]))
        axis.plot(x_values, y_values, marker="o", linewidth=1.5, label=target_id)
    for gate in DEFAULT_ACTUAL_RATIO_GATES:
        axis.axhline(gate, color="#777777", linestyle=":", linewidth=0.9)
        axis.text(max(counts) + 0.05, gate, f"{gate:.2f}", va="center", fontsize=8, color="#555555")
    axis.set_xlabel("module/control-point count")
    axis.set_ylabel("target RMSE ratio vs U10")
    axis.set_title("Actual RODM gate curves by target")
    axis.set_xticks(counts)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = sweep.FIGURE_DIR / "actual_gate_rmse_ratio_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    *,
    targets: tuple[str, ...],
    counts: tuple[int, ...],
    wavelengths_m: tuple[int, ...],
    candidate_csv: Path,
    geometry_csv: Path,
    metrics_csv: Path,
    layout_summary_csv: Path,
    gate_csv: Path,
    minimum_csv: Path,
    figures: dict[str, Path],
) -> Path:
    gate_rows = mcpv.read_csv(gate_csv)
    minimum_rows = [
        row
        for row in mcpv.read_csv(minimum_csv)
        if row["actual_ratio_gate"] in {"1.0", "0.95", "0.9"}
    ]

    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    summary_rows = []
    for target_id in targets:
        target_rows = [row for row in gate_rows if row["target_id"] == target_id]
        best = min(target_rows, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
        summary_rows.append(
            (
                target_id,
                f"N{best['module_count']}",
                best["display_name"],
                f"{float(best['actual_rmse_ratio_vs_U10']):.4f}",
                f"{float(best['actual_improvement_vs_U10_percent']):.2f}%",
            )
        )

    min_rows = [
        (
            row["target_id"],
            row["actual_ratio_gate"],
            row["minimum_module_count"] or "not passed",
            row["selected_layout_id"] or "-",
            (
                "-"
                if row["selected_actual_improvement_vs_U10_percent"] == ""
                else f"{float(row['selected_actual_improvement_vs_U10_percent']):.2f}%"
            ),
        )
        for row in minimum_rows
    ]

    lines = [
        "# 控制点数量的真实 RODM 门控研究",
        "",
        "## 1. 目的",
        "",
        (
            "本研究把密度指标生成的 N10/N11/N12/N14 候选布局接入真实水动力和 RODM 计算。"
            "注意：不同模块数量或不同模块长度对应不同 Capytaine 水动力问题，因此本脚本对每个"
            " distinct layout 单独生成 `.nc`，再进行 ordered SEREP-ridge RODM 求解。"
        ),
        "",
        f"- 目标：`{', '.join(targets)}`",
        f"- 模块数：`{', '.join('N' + str(value) for value in counts)}`",
        f"- 波长：`{', '.join(str(value) for value in wavelengths_m)} m`",
        "- 参考解：`U30 ordered SEREP-ridge`",
        "",
        "## 2. 门控图",
        "",
        f"![Actual gate heatmap]({mcpv.file_uri(figures['heatmap'])})",
        "",
        f"![Actual gate curves]({mcpv.file_uri(figures['curves'])})",
        "",
        "## 3. 每个目标下的实际最优候选",
        "",
        table(("target", "best N", "best layout", "RMSE ratio vs U10", "improvement"), summary_rows),
        "",
        "## 4. 给定真实误差门槛的最小控制点数",
        "",
        table(("target", "actual ratio gate", "minimum N", "selected layout", "improvement"), min_rows),
        "",
        "## 5. 输出文件",
        "",
        f"- 候选来源：`{candidate_csv}`",
        f"- 几何与 FEM 控制点：`{geometry_csv}`",
        f"- 逐波长误差：`{metrics_csv}`",
        f"- 布局汇总：`{layout_summary_csv}`",
        f"- 目标门控表：`{gate_csv}`",
        f"- 最小控制点表：`{minimum_csv}`",
        f"- 图目录：`{sweep.FIGURE_DIR}`",
        "",
        "## 6. 论文含义",
        "",
        (
            "这一步把“为什么需要更多控制点”的密度解释，推进成“在真实 RODM 误差门槛下"
            "需要多少控制点”的算法证据。若某些目标下 N11 或 N12 已通过，而 N14 进一步改善有限，"
            "就可以把最少控制点选择表述为一种精度-成本折中算法，而不是简单地越密越好。"
        ),
        "",
    ]
    path = sweep.OUTPUT_ROOT / "control_point_count_gate_study_report.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    targets = parse_targets(args.targets)
    counts = parse_ints(args.counts, DEFAULT_COUNTS)
    wavelengths_m = mcpv.parse_wavelengths(args.wavelengths)
    ratio_gates = parse_floats(args.actual_ratio_gates, DEFAULT_ACTUAL_RATIO_GATES)

    output_root = args.output_root or DEFAULT_OUTPUT_ROOT
    mcpv.configure_sweep_output(output_root)
    sweep.REPORT_PATH = output_root / "control_point_count_gate_study_report.md"

    candidates = load_gate_candidates(targets, counts)
    candidate_layouts = tuple(candidate.layout for candidate in candidates)
    layouts = (
        sweep.LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
        sweep.LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
        *candidate_layouts,
    )
    mcpv.update_plot_colors(layouts)

    configs = {layout.layout_id: sweep.hydro_config(layout, wavelengths_m, n_jobs=args.n_jobs) for layout in layouts}
    geometry_by_layout = {layout.layout_id: sweep.geometry_rows(layout, configs[layout.layout_id]) for layout in layouts}
    geometry_paths = {layout.layout_id: sweep.write_geometry_csv(layout, geometry_by_layout[layout.layout_id]) for layout in layouts}
    candidate_csv = write_candidate_source_table(candidates)
    geometry_csv = write_geometry_manifest(layouts, configs, geometry_paths)

    if args.dry_run:
        manifest = {
            "mode": "dry_run",
            "candidate_csv": str(candidate_csv),
            "geometry_csv": str(geometry_csv),
            "candidate_count": str(len(candidates)),
            "distinct_hydrodynamic_layouts_to_generate": str(len(candidate_layouts)),
        }
        (sweep.OUTPUT_ROOT / "control_point_count_gate_study_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    response_paths = mcpv.copy_reference_responses(wavelengths_m=wavelengths_m)
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
    gate_csv, minimum_csv = write_actual_gate_tables(
        candidates=candidates,
        targets=targets,
        counts=counts,
        wavelengths_m=wavelengths_m,
        metric_csv=metrics_csv,
        actual_ratio_gates=ratio_gates,
    )
    figures = {
        "heatmap": plot_gate_heatmap(gate_csv, targets, counts),
        "curves": plot_gate_curves(gate_csv, targets, counts),
    }
    report = write_report(
        targets=targets,
        counts=counts,
        wavelengths_m=wavelengths_m,
        candidate_csv=candidate_csv,
        geometry_csv=geometry_csv,
        metrics_csv=metrics_csv,
        layout_summary_csv=layout_summary_csv,
        gate_csv=gate_csv,
        minimum_csv=minimum_csv,
        figures=figures,
    )
    manifest = {
        "mode": "full",
        "report": str(report),
        "candidate_csv": str(candidate_csv),
        "geometry_csv": str(geometry_csv),
        "metrics_csv": str(metrics_csv),
        "layout_summary_csv": str(layout_summary_csv),
        "gate_csv": str(gate_csv),
        "minimum_csv": str(minimum_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (sweep.OUTPUT_ROOT / "control_point_count_gate_study_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=None, help="Target ids separated by comma/space.")
    parser.add_argument("--counts", default=None, help="Module counts separated by comma/space. Default: 10 11 12 14.")
    parser.add_argument("--wavelengths", default=None, help="Wavelengths in m. Default: 60 120 180 240 300.")
    parser.add_argument("--actual-ratio-gates", default=None, help="RMSE-ratio gates vs U10. Default: 1.0 0.95 0.9.")
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
