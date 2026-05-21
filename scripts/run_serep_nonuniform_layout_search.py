"""Search non-uniform 10-module layouts for ordered SEREP-ridge RODM.

The search is intentionally two-stage. It first enumerates all 10-module
layouts using only 20/30/40 m modules with total length 300 m, ranks them with
a cheap U30-reference sampling surrogate, and then runs full Capytaine + RODM
only for a small selected set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_serep_nonuniform_design_study as base  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_layout_search"
PREVIOUS_STUDY_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_design_study"
MODULE_LENGTH_CHOICES_M = (20.0, 30.0, 40.0)
MODULE_COUNT = 10
TOTAL_LENGTH_M = 300.0


FORCED_LAYOUTS: dict[str, tuple[float, ...]] = {
    "prev_bow_refined": (20, 20, 20, 30, 30, 30, 30, 40, 40, 40),
    "prev_center_refined": (40, 40, 30, 20, 20, 20, 20, 30, 40, 40),
    "prev_edge_mild": (20, 30, 30, 40, 30, 30, 40, 30, 30, 20),
}


PREVIOUS_HYDRO_BY_LENGTHS = {
    tuple(float(value) for value in lengths): PREVIOUS_STUDY_ROOT
    / "hydro"
    / f"{layout_id}_U10_D0p5_rho1000_wl60_300_mesh2.nc"
    for layout_id, lengths in base.LAYOUTS.items()
}


@dataclass(frozen=True)
class Candidate:
    layout_id: str
    lengths_m: tuple[float, ...]
    pair_count_20_40: int
    surrogate_mean_rmse: float
    surrogate_max_rmse: float
    surrogate_rmse_by_wavelength: tuple[float, ...]
    selection_reason: str


def length_signature(lengths_m: tuple[float, ...]) -> str:
    return "".join(str(int(value / 10.0)) for value in lengths_m)


def module_boundaries(lengths_m: tuple[float, ...]) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])


def module_centers(lengths_m: tuple[float, ...]) -> np.ndarray:
    boundaries = module_boundaries(lengths_m)
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def enumerate_layouts() -> list[tuple[float, ...]]:
    layouts = []
    for values in product(MODULE_LENGTH_CHOICES_M, repeat=MODULE_COUNT):
        if math.isclose(sum(values), TOTAL_LENGTH_M, rel_tol=0.0, abs_tol=1.0e-9):
            layouts.append(tuple(float(value) for value in values))
    return layouts


def reference_heave_by_wavelength() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    references = {}
    for wavelength_m in base.WAVELENGTHS_M:
        response = np.load(base.reference_response_path(wavelength_m))
        references[int(wavelength_m)] = extract_centerline_heave(response)
    return references


def piecewise_constant_surrogate(
    lengths_m: tuple[float, ...],
    references: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, tuple[float, ...]]:
    boundaries = module_boundaries(lengths_m) / TOTAL_LENGTH_M
    centers = module_centers(lengths_m) / TOTAL_LENGTH_M
    errors = []
    for _wavelength_m, (x_ref, heave_ref) in references.items():
        sampled = np.interp(centers, x_ref, heave_ref)
        approx = np.empty_like(heave_ref)
        for index, value in enumerate(heave_ref):
            segment = int(np.searchsorted(boundaries, x_ref[index], side="right") - 1)
            segment = max(0, min(segment, len(sampled) - 1))
            approx[index] = sampled[segment]
        errors.append(float(np.sqrt(np.mean((approx - heave_ref) ** 2))))
    return float(np.mean(errors)), float(np.max(errors)), tuple(errors)


def build_candidate_table() -> list[Candidate]:
    references = reference_heave_by_wavelength()
    candidates = []
    for lengths in enumerate_layouts():
        if lengths == (30.0,) * MODULE_COUNT:
            continue
        mean_rmse, max_rmse, by_wavelength = piecewise_constant_surrogate(lengths, references)
        pair_count = sum(1 for value in lengths if value == 20.0)
        candidates.append(
            Candidate(
                layout_id=f"cand_{length_signature(lengths)}",
                lengths_m=lengths,
                pair_count_20_40=pair_count,
                surrogate_mean_rmse=mean_rmse,
                surrogate_max_rmse=max_rmse,
                surrogate_rmse_by_wavelength=by_wavelength,
                selection_reason="",
            )
        )
    return sorted(candidates, key=lambda item: (item.surrogate_mean_rmse, item.surrogate_max_rmse))


def hamming_distance(left: tuple[float, ...], right: tuple[float, ...]) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def candidate_with_reason(candidate: Candidate, reason: str) -> Candidate:
    return Candidate(
        layout_id=candidate.layout_id,
        lengths_m=candidate.lengths_m,
        pair_count_20_40=candidate.pair_count_20_40,
        surrogate_mean_rmse=candidate.surrogate_mean_rmse,
        surrogate_max_rmse=candidate.surrogate_max_rmse,
        surrogate_rmse_by_wavelength=candidate.surrogate_rmse_by_wavelength,
        selection_reason=reason,
    )


def select_candidates(
    candidates: list[Candidate],
    *,
    evaluate_count: int,
    diversity_min_hamming: int,
) -> list[Candidate]:
    by_lengths = {candidate.lengths_m: candidate for candidate in candidates}
    selected: list[Candidate] = []
    selected_lengths: set[tuple[float, ...]] = set()

    for forced_id, lengths in FORCED_LAYOUTS.items():
        candidate = by_lengths[tuple(float(value) for value in lengths)]
        selected.append(
            Candidate(
                layout_id=forced_id,
                lengths_m=candidate.lengths_m,
                pair_count_20_40=candidate.pair_count_20_40,
                surrogate_mean_rmse=candidate.surrogate_mean_rmse,
                surrogate_max_rmse=candidate.surrogate_max_rmse,
                surrogate_rmse_by_wavelength=candidate.surrogate_rmse_by_wavelength,
                selection_reason="previous-best-layout",
            )
        )
        selected_lengths.add(candidate.lengths_m)

    for pair_count in range(1, 6):
        if len(selected) >= evaluate_count:
            break
        group = [item for item in candidates if item.pair_count_20_40 == pair_count]
        for candidate in group:
            if candidate.lengths_m in selected_lengths:
                continue
            selected.append(candidate_with_reason(candidate, f"best-surrogate-pair-count-{pair_count}"))
            selected_lengths.add(candidate.lengths_m)
            break

    for candidate in candidates:
        if len(selected) >= evaluate_count:
            break
        if candidate.lengths_m in selected_lengths:
            continue
        if any(hamming_distance(candidate.lengths_m, item.lengths_m) < diversity_min_hamming for item in selected):
            continue
        selected.append(candidate_with_reason(candidate, "global-surrogate-diverse"))
        selected_lengths.add(candidate.lengths_m)

    for candidate in candidates:
        if len(selected) >= evaluate_count:
            break
        if candidate.lengths_m in selected_lengths:
            continue
        selected.append(candidate_with_reason(candidate, "global-surrogate-fill"))
        selected_lengths.add(candidate.lengths_m)

    return selected[:evaluate_count]


def candidate_hydro_config(candidate: Candidate, *, n_jobs: int) -> base.ArrayHydrodynamicsConfig:
    config = base.build_hydro_config(candidate.layout_id, candidate.lengths_m, n_jobs=n_jobs)
    previous_hydro = PREVIOUS_HYDRO_BY_LENGTHS.get(candidate.lengths_m)
    if previous_hydro is not None and previous_hydro.exists():
        return replace(config, output_path=previous_hydro)
    return config


def is_under_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return False
    return True


def write_candidate_tables(
    *,
    candidates: list[Candidate],
    selected: list[Candidate],
) -> tuple[Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preselection_path = OUTPUT_ROOT / "serep_nonuniform_candidate_preselection.csv"
    selected_path = OUTPUT_ROOT / "serep_nonuniform_selected_candidates.csv"
    fieldnames = [
        "rank",
        "layout_id",
        "lengths_m",
        "pair_count_20_40",
        "surrogate_mean_rmse",
        "surrogate_max_rmse",
        "surrogate_rmse_60m",
        "surrogate_rmse_120m",
        "surrogate_rmse_180m",
        "surrogate_rmse_240m",
        "surrogate_rmse_300m",
        "selection_reason",
    ]
    selected_by_lengths = {item.lengths_m: item.selection_reason for item in selected}
    with preselection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            row = candidate_row(candidate, rank=rank)
            row["selection_reason"] = selected_by_lengths.get(candidate.lengths_m, "")
            writer.writerow(row)
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(selected, start=1):
            writer.writerow(candidate_row(candidate, rank=rank))
    return preselection_path, selected_path


def candidate_row(candidate: Candidate, *, rank: int) -> dict[str, object]:
    row = {
        "rank": rank,
        "layout_id": candidate.layout_id,
        "lengths_m": " ".join(f"{value:g}" for value in candidate.lengths_m),
        "pair_count_20_40": candidate.pair_count_20_40,
        "surrogate_mean_rmse": candidate.surrogate_mean_rmse,
        "surrogate_max_rmse": candidate.surrogate_max_rmse,
        "selection_reason": candidate.selection_reason,
    }
    for wavelength_m, value in zip(base.WAVELENGTHS_M, candidate.surrogate_rmse_by_wavelength):
        row[f"surrogate_rmse_{wavelength_m}m"] = value
    return row


def solve_selected_candidates(
    selected: list[Candidate],
    *,
    force_hydro: bool,
    n_jobs: int,
) -> tuple[list[base.CaseResult], dict[str, Path], dict[str, tuple[float, ...]]]:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    results = base.add_uniform_baseline()
    geometry_paths: dict[str, Path] = {
        "uniform_U10": base.write_geometry_csv(
            "uniform_U10",
            base.build_hydro_config("uniform_U10", (30.0,) * 10, n_jobs=n_jobs),
        ),
        "uniform_U30_reference": base.write_geometry_csv(
            "uniform_U30_reference",
            base.build_hydro_config("uniform_U30_reference", (10.0,) * 30, n_jobs=n_jobs),
        ),
    }
    layouts: dict[str, tuple[float, ...]] = {"uniform_U10": (30.0,) * 10}
    for candidate in selected:
        config = candidate_hydro_config(candidate, n_jobs=n_jobs)
        base.ensure_hydrodynamics(config, force=force_hydro and is_under_output_root(config.output_path))
        geometry_paths[candidate.layout_id] = base.write_geometry_csv(candidate.layout_id, config)
        results.extend(base.solve_layout(candidate.layout_id, config))
        layouts[candidate.layout_id] = candidate.lengths_m
    return results, geometry_paths, layouts


def ranked_actual_results(results: list[base.CaseResult]) -> list[dict[str, object]]:
    layout_ids = list(dict.fromkeys(item.layout_id for item in results))
    rows = []
    for layout_id in layout_ids:
        layout_rows = [item for item in results if item.layout_id == layout_id]
        rows.append(
            {
                "layout_id": layout_id,
                "mean_rmse": float(np.mean([item.rmse_vs_U30_serep_ridge for item in layout_rows])),
                "mean_max_abs": float(np.mean([item.max_abs_vs_U30_serep_ridge for item in layout_rows])),
            }
        )
    return sorted(rows, key=lambda item: item["mean_rmse"])


def write_actual_summary(
    *,
    results: list[base.CaseResult],
    selected: list[Candidate],
    geometry_paths: dict[str, Path],
    layouts: dict[str, tuple[float, ...]],
    response_panel: Path,
    error_summary: Path,
    preselection_path: Path,
    selected_path: Path,
) -> tuple[Path, Path, Path]:
    summary_path = OUTPUT_ROOT / "serep_nonuniform_layout_search_summary.csv"
    ranking_path = OUTPUT_ROOT / "serep_nonuniform_layout_search_ranking.csv"
    report_path = OUTPUT_ROOT / "serep_nonuniform_layout_search_report.md"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(base.CaseResult.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = item.__dict__.copy()
            for key in ("response_path", "hydro_path", "figure_path"):
                row[key] = str(row[key])
            writer.writerow(row)

    ranking = ranked_actual_results(results)
    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "layout_id", "mean_rmse", "mean_max_abs", "lengths_m"])
        writer.writeheader()
        for rank, row in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                    "lengths_m": " ".join(f"{value:g}" for value in layouts.get(row["layout_id"], ())),
                }
            )

    lines = [
        "# SEREP-ridge Non-uniform Layout Search",
        "",
        "All full solves use ordered SEREP-ridge and compare heave against U30 SEREP-ridge.",
        "",
        f"- preselection CSV: `{preselection_path}`",
        f"- selected candidates CSV: `{selected_path}`",
        f"- full-solve summary CSV: `{summary_path}`",
        f"- full-solve ranking CSV: `{ranking_path}`",
        f"- response panel: `{response_panel}`",
        f"- error summary: `{error_summary}`",
        "",
        "## Full-solve ranking",
        "",
        "| rank | layout | mean RMSE | mean max abs | lengths m |",
        "| ---: | :--- | ---: | ---: | :--- |",
    ]
    for rank, row in enumerate(ranking, start=1):
        lengths = layouts.get(row["layout_id"], ())
        lines.append(
            f"| {rank} | {row['layout_id']} | {row['mean_rmse']:.9g} | "
            f"{row['mean_max_abs']:.9g} | `{list(lengths)}` |"
        )
    lines.extend(
        [
            "",
            "## Selected candidate surrogate scores",
            "",
            "| layout | reason | surrogate mean RMSE | lengths m |",
            "| :--- | :--- | ---: | :--- |",
        ]
    )
    for candidate in selected:
        lines.append(
            f"| {candidate.layout_id} | {candidate.selection_reason} | "
            f"{candidate.surrogate_mean_rmse:.9g} | `{list(candidate.lengths_m)}` |"
        )
    lines.extend(
        [
            "",
            "## Per-wavelength RMSE",
            "",
            "| layout | 60 | 120 | 180 | 240 | 300 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layout_id in list(dict.fromkeys(item.layout_id for item in results)):
        rows = [item for item in results if item.layout_id == layout_id]
        lines.append(
            f"| {layout_id} | "
            + " | ".join(f"{item.rmse_vs_U30_serep_ridge:.9g}" for item in rows)
            + " |"
        )
    lines.extend(["", "## Geometry tables", ""])
    for layout_id, path in geometry_paths.items():
        lines.append(f"- {layout_id}: `{path}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "parameters": {
            "method": "serep_ridge",
            "module_length_choices_m": list(MODULE_LENGTH_CHOICES_M),
            "module_count": MODULE_COUNT,
            "total_length_m": TOTAL_LENGTH_M,
            "wavelengths_m": list(base.WAVELENGTHS_M),
        },
        "selected": [
            {
                "layout_id": item.layout_id,
                "lengths_m": list(item.lengths_m),
                "selection_reason": item.selection_reason,
                "surrogate_mean_rmse": item.surrogate_mean_rmse,
            }
            for item in selected
        ],
        "ranking_csv": str(ranking_path),
        "summary_csv": str(summary_path),
        "report": str(report_path),
        "geometry_csv": {key: str(value) for key, value in geometry_paths.items()},
        "response_panel": str(response_panel),
        "error_summary": str(error_summary),
    }
    (OUTPUT_ROOT / "serep_nonuniform_layout_search_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path, ranking_path, report_path


def plot_response_panel(results: list[base.CaseResult], layouts: dict[str, tuple[float, ...]]) -> Path:
    import matplotlib.pyplot as plt

    ranking = ranked_actual_results(results)
    top_nonuniform = [row["layout_id"] for row in ranking if row["layout_id"] != "uniform_U10"][:5]
    plot_layout_ids = ["uniform_U10", *top_nonuniform]
    response_by_layout_wl = {(item.layout_id, item.wavelength_m): item.response_path for item in results}
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_layout_ids)))
    color_by_layout = dict(zip(plot_layout_ids, colors))

    fig, axes = plt.subplots(len(base.WAVELENGTHS_M), 1, figsize=(11.0, 16.0), sharex=True)
    fig.suptitle("SEREP-ridge Layout Search: Best Non-uniform Heave vs U30", fontsize=16)
    for axis, wavelength_m in zip(axes, base.WAVELENGTHS_M):
        reference = np.load(base.reference_response_path(wavelength_m))
        x, heave_ref = extract_centerline_heave(reference)
        axis.plot(x, heave_ref, color="#111111", linewidth=2.2, label="U30 reference")
        for layout_id in plot_layout_ids:
            response = np.load(response_by_layout_wl[(layout_id, wavelength_m)])
            _, heave = extract_centerline_heave(response)
            axis.plot(
                x,
                heave,
                linewidth=1.4,
                linestyle="-" if layout_id == "uniform_U10" else "--",
                color=color_by_layout[layout_id],
                label=layout_id,
            )
        axis.set_ylabel(f"{wavelength_m} m\nHeave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("x/L")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = OUTPUT_ROOT / "figures" / "serep_nonuniform_layout_search_heave_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_error_summary(results: list[base.CaseResult]) -> Path:
    import matplotlib.pyplot as plt

    ranking = ranked_actual_results(results)
    layout_ids = [row["layout_id"] for row in ranking]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(layout_ids))
    axes[0].bar(x, [row["mean_rmse"] for row in ranking], color="#1f77b4")
    axes[0].set_title("Mean RMSE Across Wavelengths")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layout_ids, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE")

    for layout_id in layout_ids[:8]:
        rows = [item for item in results if item.layout_id == layout_id]
        axes[1].plot(
            base.WAVELENGTHS_M,
            [item.rmse_vs_U30_serep_ridge for item in rows],
            marker="o",
            linewidth=1.4,
            label=layout_id,
        )
    axes[1].set_title("RMSE by Wavelength")
    axes[1].set_xlabel("wavelength (m)")
    axes[1].set_xticks(base.WAVELENGTHS_M)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    fig.suptitle("SEREP-ridge Layout Search Error Ranking", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT_ROOT / "figures" / "serep_nonuniform_layout_search_error_ranking.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run_workflow(args: argparse.Namespace) -> None:
    candidates = build_candidate_table()
    selected = select_candidates(
        candidates,
        evaluate_count=args.evaluate_count,
        diversity_min_hamming=args.diversity_min_hamming,
    )
    preselection_path, selected_path = write_candidate_tables(candidates=candidates, selected=selected)
    if args.dry_run:
        print(f"preselection={preselection_path}")
        print(f"selected={selected_path}")
        return

    results, geometry_paths, layouts = solve_selected_candidates(
        selected,
        force_hydro=args.force_hydro,
        n_jobs=args.n_jobs,
    )
    response_panel = plot_response_panel(results, layouts)
    error_summary = plot_error_summary(results)
    summary_path, ranking_path, report_path = write_actual_summary(
        results=results,
        selected=selected,
        geometry_paths=geometry_paths,
        layouts=layouts,
        response_panel=response_panel,
        error_summary=error_summary,
        preselection_path=preselection_path,
        selected_path=selected_path,
    )
    print(f"summary={summary_path}")
    print(f"ranking={ranking_path}")
    print(f"report={report_path}")
    print(f"response_panel={response_panel}")
    print(f"error_summary={error_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-count", type=int, default=12, help="Number of non-uniform layouts to solve fully.")
    parser.add_argument("--diversity-min-hamming", type=int, default=3, help="Minimum Hamming distance for global picks.")
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate hydrodynamic files in this search folder.")
    parser.add_argument("--n-jobs", type=int, default=base.CAPYTAINE_N_JOBS, help="Capytaine worker count.")
    parser.add_argument("--dry-run", action="store_true", help="Only write candidate preselection tables.")
    return parser.parse_args()


def main() -> int:
    run_workflow(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
