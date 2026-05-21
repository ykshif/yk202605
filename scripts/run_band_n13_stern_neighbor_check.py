"""Check a small N13 stern-refined neighborhood for the band target.

The targeted refinement found an N13 layout with band_equal_120_300m RMSE
ratio 0.90088, just above the 0.90 gate.  This script checks a tiny local
neighborhood: keep N13, keep 20/30 m modules, and choose which four of the
last six module positions are 30 m.
"""

from __future__ import annotations

from dataclasses import replace
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
import run_targeted_nonuniform_refinement as targeted  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "band_n13_stern_neighbor_check"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
WAVELENGTHS_M = (60, 120, 180, 240, 300)
BAND_WAVELENGTHS_M = (120, 180, 240, 300)


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def module_centers(lengths_m: tuple[float, ...]) -> np.ndarray:
    boundaries = np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def lengths_from_30_positions(positions: tuple[int, ...]) -> tuple[float, ...]:
    lengths = [20.0 for _ in range(13)]
    for position in positions:
        lengths[position] = 30.0
    if not np.isclose(sum(lengths), 300.0):
        raise ValueError(lengths)
    return tuple(lengths)


def build_layouts() -> tuple[sweep.LayoutSpec, ...]:
    layouts = []
    for index, positions in enumerate(combinations(range(7, 13), 4), start=1):
        lengths = lengths_from_30_positions(tuple(positions))
        tag = "".join(f"{position:02d}" for position in positions)
        layouts.append(
            sweep.LayoutSpec(
                layout_id=f"BN13_{tag}",
                display_name=f"N13 stern {tag}",
                category="band_n13_stern_neighbor",
                module_lengths_m=lengths,
            )
        )
    return tuple(layouts)


def geometry_manifest(
    layouts: tuple[sweep.LayoutSpec, ...],
    configs: dict[str, sweep.ArrayHydrodynamicsConfig],
    geometry_paths: dict[str, Path],
) -> Path:
    rows = []
    for layout in layouts:
        geometry = sweep.geometry_rows(layout, configs[layout.layout_id])
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(layout.module_lengths_m)),
                "selected_node_ids": " ".join(str(int(row["selected_node_id"])) for row in geometry),
                "max_abs_center_node_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "hydro_path": str(configs[layout.layout_id].output_path),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
            }
        )
    return write_csv(TABLE_DIR / "band_n13_stern_neighbor_geometry.csv", rows)


def mean_rmse(metric_rows: list[dict[str, str]], layout_id: str, wavelengths_m: tuple[int, ...]) -> float:
    return float(np.mean([mcpv.metric_lookup(metric_rows, layout_id, wavelength, "rmse_vs_U30") for wavelength in wavelengths_m]))


def actual_table(layouts: tuple[sweep.LayoutSpec, ...], metric_csv: Path) -> Path:
    metric_rows = mcpv.read_csv(metric_csv)
    u10_rmse = mean_rmse(metric_rows, "uniform_U10", BAND_WAVELENGTHS_M)
    rows = []
    for layout in layouts:
        if layout.layout_id == "U30_reference":
            continue
        actual_rmse = mean_rmse(metric_rows, layout.layout_id, BAND_WAVELENGTHS_M)
        ratio = actual_rmse / u10_rmse
        rows.append(
            {
                "layout_id": layout.layout_id,
                "module_count": layout.module_count,
                "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
                "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(layout.module_lengths_m)),
                "actual_rmse_vs_U30": actual_rmse,
                "uniform_U10_rmse_vs_U30": u10_rmse,
                "actual_rmse_ratio_vs_U10": ratio,
                "actual_improvement_vs_U10_percent": (1.0 - ratio) * 100.0,
            }
        )
    rows.sort(key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
    return write_csv(TABLE_DIR / "band_n13_stern_neighbor_actual.csv", rows)


def plot_results(actual_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(actual_csv)
    labels = [row["layout_id"].replace("BN13_", "") for row in rows]
    values = [float(row["actual_rmse_ratio_vs_U10"]) for row in rows]
    fig, axis = plt.subplots(figsize=(11.4, 4.8))
    axis.bar(np.arange(len(values)), values, color="#4c78a8")
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, label="U10")
    axis.axhline(0.9, color="#d62728", linestyle="--", linewidth=1.0, label="0.90 gate")
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("band RMSE ratio vs U10")
    axis.set_title("N13 stern-neighborhood check for band_equal_120_300m")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "band_n13_stern_neighbor_ratio.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    mcpv.configure_sweep_output(OUTPUT_ROOT)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    candidate_layouts = build_layouts()
    layouts = (
        sweep.LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
        sweep.LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
        *candidate_layouts,
    )
    previous_hydro = targeted.previous_hydro_by_lengths()
    configs = {}
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
    geometry_csv = geometry_manifest(layouts, configs, geometry_paths)

    if args.dry_run:
        manifest = {"mode": "dry_run", "geometry_csv": str(geometry_csv), "candidate_count": str(len(candidate_layouts))}
        (OUTPUT_ROOT / "band_n13_stern_neighbor_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    response_paths = mcpv.copy_reference_responses(wavelengths_m=WAVELENGTHS_M)
    for layout in candidate_layouts:
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
    actual_csv = actual_table(layouts, metric_csv)
    figure = plot_results(actual_csv)
    manifest = {
        "mode": "full",
        "geometry_csv": str(geometry_csv),
        "metric_csv": str(metric_csv),
        "layout_summary_csv": str(layout_summary_csv),
        "actual_csv": str(actual_csv),
        "figure": str(figure),
    }
    (OUTPUT_ROOT / "band_n13_stern_neighbor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
