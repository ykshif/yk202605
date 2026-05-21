"""Refine target-wavelength non-uniform layouts for SEREP-ridge RODM.

This script continues the previous 20/30/40 m layout search. It reads the
existing full-solve results, identifies the best layout for each target
wavelength, then evaluates nearby layouts instead of starting from scratch.
"""

from __future__ import annotations

from dataclasses import replace
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

import run_serep_nonuniform_design_study as base  # noqa: E402
import run_serep_nonuniform_layout_search as search  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_target_refinement"
PREVIOUS_SEARCH_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_layout_search"
PREVIOUS_SUMMARY = PREVIOUS_SEARCH_ROOT / "serep_nonuniform_layout_search_summary.csv"
PREVIOUS_RANKING = PREVIOUS_SEARCH_ROOT / "serep_nonuniform_layout_search_ranking.csv"
TARGET_WAVELENGTHS_M = (120, 180, 240, 300)


def parse_lengths(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.replace(",", " ").split() if value)


def load_previous_layouts() -> dict[str, tuple[float, ...]]:
    layouts: dict[str, tuple[float, ...]] = {}
    with PREVIOUS_RANKING.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layouts[row["layout_id"]] = parse_lengths(row["lengths_m"])
    return layouts


def load_previous_results() -> list[base.CaseResult]:
    results: list[base.CaseResult] = []
    with PREVIOUS_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            results.append(
                base.CaseResult(
                    layout_id=row["layout_id"],
                    wavelength_m=int(row["wavelength_m"]),
                    omega_rad_s=float(row["omega_rad_s"]),
                    rmse_vs_U30_serep_ridge=float(row["rmse_vs_U30_serep_ridge"]),
                    max_abs_vs_U30_serep_ridge=float(row["max_abs_vs_U30_serep_ridge"]),
                    roughness=float(row["roughness"]),
                    response_path=Path(row["response_path"]),
                    hydro_path=Path(row["hydro_path"]),
                    figure_path=Path(row["figure_path"]),
                )
            )
    return results


def previous_hydro_by_lengths(
    previous_results: list[base.CaseResult],
    layouts: dict[str, tuple[float, ...]],
) -> dict[tuple[float, ...], Path]:
    mapping: dict[tuple[float, ...], Path] = {}
    for result in previous_results:
        lengths = layouts.get(result.layout_id)
        if lengths is not None and result.hydro_path.exists():
            mapping[lengths] = result.hydro_path
    return mapping


def previous_geometry_paths(layouts: dict[str, tuple[float, ...]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for layout_id in layouts:
        path = PREVIOUS_SEARCH_ROOT / "geometry" / f"{layout_id}_module_geometry.csv"
        if path.exists():
            paths[layout_id] = path
    u30_path = PREVIOUS_SEARCH_ROOT / "geometry" / "uniform_U30_reference_module_geometry.csv"
    if u30_path.exists():
        paths["uniform_U30_reference"] = u30_path
    return paths


def best_layout_by_wavelength(
    previous_results: list[base.CaseResult],
) -> dict[int, base.CaseResult]:
    best: dict[int, base.CaseResult] = {}
    for wavelength_m in TARGET_WAVELENGTHS_M:
        rows = [item for item in previous_results if item.wavelength_m == wavelength_m]
        best[wavelength_m] = min(rows, key=lambda item: item.rmse_vs_U30_serep_ridge)
    return best


def best_nonuniform_mean_layout(
    previous_results: list[base.CaseResult],
) -> str:
    layout_ids = sorted({item.layout_id for item in previous_results if item.layout_id != "uniform_U10"})
    scored = []
    for layout_id in layout_ids:
        rows = [item for item in previous_results if item.layout_id == layout_id]
        scored.append((float(np.mean([item.rmse_vs_U30_serep_ridge for item in rows])), layout_id))
    return min(scored)[1]


def candidate_from_existing(
    candidate: search.Candidate,
    *,
    layout_id: str,
    reason: str,
) -> search.Candidate:
    return search.Candidate(
        layout_id=layout_id,
        lengths_m=candidate.lengths_m,
        pair_count_20_40=candidate.pair_count_20_40,
        surrogate_mean_rmse=candidate.surrogate_mean_rmse,
        surrogate_max_rmse=candidate.surrogate_max_rmse,
        surrogate_rmse_by_wavelength=candidate.surrogate_rmse_by_wavelength,
        selection_reason=reason,
    )


def select_refinement_candidates(
    *,
    previous_results: list[base.CaseResult],
    layouts: dict[str, tuple[float, ...]],
    per_target_count: int,
    mean_count: int,
    local_hamming: int,
) -> list[search.Candidate]:
    evaluated_lengths = set(layouts.values())
    all_candidates = search.build_candidate_table()
    by_wavelength_index = {int(value): index for index, value in enumerate(base.WAVELENGTHS_M)}
    selected: list[search.Candidate] = []
    selected_lengths: set[tuple[float, ...]] = set()

    for wavelength_m, best_result in best_layout_by_wavelength(previous_results).items():
        best_lengths = layouts[best_result.layout_id]
        index = by_wavelength_index[wavelength_m]
        pool = [
            item
            for item in all_candidates
            if item.lengths_m not in evaluated_lengths
            and item.lengths_m not in selected_lengths
            and search.hamming_distance(item.lengths_m, best_lengths) <= local_hamming
        ]
        pool.sort(
            key=lambda item: (
                item.surrogate_rmse_by_wavelength[index],
                search.hamming_distance(item.lengths_m, best_lengths),
                item.surrogate_mean_rmse,
            )
        )
        for candidate in pool[:per_target_count]:
            layout_id = f"refine_{wavelength_m}m_{search.length_signature(candidate.lengths_m)}"
            selected.append(
                candidate_from_existing(
                    candidate,
                    layout_id=layout_id,
                    reason=f"local-neighbor-of-{best_result.layout_id}-for-{wavelength_m}m",
                )
            )
            selected_lengths.add(candidate.lengths_m)

    best_mean_id = best_nonuniform_mean_layout(previous_results)
    best_mean_lengths = layouts[best_mean_id]
    mean_pool = [
        item
        for item in all_candidates
        if item.lengths_m not in evaluated_lengths
        and item.lengths_m not in selected_lengths
        and search.hamming_distance(item.lengths_m, best_mean_lengths) <= local_hamming
    ]
    mean_pool.sort(key=lambda item: (item.surrogate_mean_rmse, search.hamming_distance(item.lengths_m, best_mean_lengths)))
    for candidate in mean_pool[:mean_count]:
        layout_id = f"refine_mean_{search.length_signature(candidate.lengths_m)}"
        selected.append(
            candidate_from_existing(
                candidate,
                layout_id=layout_id,
                reason=f"local-neighbor-of-{best_mean_id}-for-mean-rmse",
            )
        )
        selected_lengths.add(candidate.lengths_m)

    return selected


def write_selected_table(selected: list[search.Candidate]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "serep_target_refinement_selected_candidates.csv"
    fieldnames = [
        "rank",
        "layout_id",
        "lengths_m",
        "pair_count_20_40",
        "surrogate_mean_rmse",
        "surrogate_rmse_60m",
        "surrogate_rmse_120m",
        "surrogate_rmse_180m",
        "surrogate_rmse_240m",
        "surrogate_rmse_300m",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(selected, start=1):
            row = search.candidate_row(candidate, rank=rank)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def candidate_hydro_config(
    candidate: search.Candidate,
    *,
    hydro_lookup: dict[tuple[float, ...], Path],
    n_jobs: int,
) -> base.ArrayHydrodynamicsConfig:
    config = base.build_hydro_config(candidate.layout_id, candidate.lengths_m, n_jobs=n_jobs)
    hydro_path = hydro_lookup.get(candidate.lengths_m)
    if hydro_path is not None and hydro_path.exists():
        return replace(config, output_path=hydro_path)
    return config


def solve_selected(
    selected: list[search.Candidate],
    *,
    previous_results: list[base.CaseResult],
    layouts: dict[str, tuple[float, ...]],
    hydro_lookup: dict[tuple[float, ...], Path],
    force_hydro: bool,
    n_jobs: int,
) -> tuple[list[base.CaseResult], dict[str, Path], dict[str, tuple[float, ...]]]:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    search.OUTPUT_ROOT = OUTPUT_ROOT
    results = list(previous_results)
    geometry_paths = previous_geometry_paths(layouts)
    combined_layouts = dict(layouts)
    for candidate in selected:
        config = candidate_hydro_config(candidate, hydro_lookup=hydro_lookup, n_jobs=n_jobs)
        base.ensure_hydrodynamics(config, force=force_hydro and search.is_under_output_root(config.output_path))
        geometry_paths[candidate.layout_id] = base.write_geometry_csv(candidate.layout_id, config)
        results.extend(base.solve_layout(candidate.layout_id, config))
        combined_layouts[candidate.layout_id] = candidate.lengths_m
    return results, geometry_paths, combined_layouts


def write_target_comparison(results: list[base.CaseResult]) -> Path:
    path = OUTPUT_ROOT / "serep_target_refinement_best_by_wavelength.csv"
    rows = []
    for wavelength_m in base.WAVELENGTHS_M:
        wl_rows = [item for item in results if item.wavelength_m == wavelength_m]
        best = min(wl_rows, key=lambda item: item.rmse_vs_U30_serep_ridge)
        uniform = next(item for item in wl_rows if item.layout_id == "uniform_U10")
        rows.append(
            {
                "wavelength_m": wavelength_m,
                "best_layout_id": best.layout_id,
                "best_rmse": best.rmse_vs_U30_serep_ridge,
                "uniform_U10_rmse": uniform.rmse_vs_U30_serep_ridge,
                "rmse_improvement_vs_uniform": uniform.rmse_vs_U30_serep_ridge
                - best.rmse_vs_U30_serep_ridge,
                "best_max_abs": best.max_abs_vs_U30_serep_ridge,
                "uniform_U10_max_abs": uniform.max_abs_vs_U30_serep_ridge,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_workflow(args: argparse.Namespace) -> None:
    previous_results = load_previous_results()
    layouts = load_previous_layouts()
    hydro_lookup = previous_hydro_by_lengths(previous_results, layouts)
    selected = select_refinement_candidates(
        previous_results=previous_results,
        layouts=layouts,
        per_target_count=args.per_target_count,
        mean_count=args.mean_count,
        local_hamming=args.local_hamming,
    )
    selected_path = write_selected_table(selected)
    if args.dry_run:
        print(f"selected={selected_path}")
        return

    results, geometry_paths, combined_layouts = solve_selected(
        selected,
        previous_results=previous_results,
        layouts=layouts,
        hydro_lookup=hydro_lookup,
        force_hydro=args.force_hydro,
        n_jobs=args.n_jobs,
    )
    response_panel = search.plot_response_panel(results, combined_layouts)
    error_summary = search.plot_error_summary(results)
    summary_path, ranking_path, report_path = search.write_actual_summary(
        results=results,
        selected=selected,
        geometry_paths=geometry_paths,
        layouts=combined_layouts,
        response_panel=response_panel,
        error_summary=error_summary,
        preselection_path=PREVIOUS_SEARCH_ROOT / "serep_nonuniform_candidate_preselection.csv",
        selected_path=selected_path,
    )
    target_path = write_target_comparison(results)
    manifest = {
        "selected_candidates": str(selected_path),
        "summary": str(summary_path),
        "ranking": str(ranking_path),
        "report": str(report_path),
        "target_comparison": str(target_path),
        "response_panel": str(response_panel),
        "error_summary": str(error_summary),
    }
    (OUTPUT_ROOT / "serep_target_refinement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selected={selected_path}")
    print(f"summary={summary_path}")
    print(f"ranking={ranking_path}")
    print(f"target_comparison={target_path}")
    print(f"report={report_path}")
    print(f"response_panel={response_panel}")
    print(f"error_summary={error_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-target-count", type=int, default=3)
    parser.add_argument("--mean-count", type=int, default=3)
    parser.add_argument("--local-hamming", type=int, default=4)
    parser.add_argument("--force-hydro", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=base.CAPYTAINE_N_JOBS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    run_workflow(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
