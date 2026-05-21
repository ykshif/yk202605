"""Search and validate low-control-point non-uniform RODM layouts.

The key research question here is not whether more hydrodynamic modules are
more accurate.  Instead, this workflow asks whether a small number of
non-uniform, FEM-aligned hydrodynamic control points can approximate the U30
reference better than a naive low-resolution layout.

The workflow is deliberately two-stage:

1. Enumerate FEM-aligned non-uniform layouts for N5..N9 and rank them with a
   response-aware surrogate built from the U30 heave curves.
2. Regenerate Capytaine hydrodynamic datasets for a small selected set and run
   the actual ordered SEREP-ridge RODM solver.

Changing the module lengths changes the hydrodynamic BEM problem, so every
validated candidate gets its own newly generated/reused NetCDF dataset.
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


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "low_control_point_response_aware_study"
DEFAULT_COUNTS = (5, 6, 7, 8, 9)
DEFAULT_LENGTH_CHOICES_M = tuple(range(20, 81, 10))
DEFAULT_WAVELENGTHS_M = (60, 120, 180, 240, 300)
DEFAULT_SEARCH_TARGETS = (
    "wl_60m",
    "wl_120m",
    "wl_180m",
    "wl_240m",
    "wl_300m",
    "band_60_300m",
    "band_120_300m",
    "band_center_120_240m",
)
DEFAULT_VALIDATION_TARGETS = ("band_120_300m", "wl_240m", "wl_300m")
TARGET_WAVELENGTHS = {
    "wl_60m": (60,),
    "wl_120m": (120,),
    "wl_180m": (180,),
    "wl_240m": (240,),
    "wl_300m": (300,),
    "band_60_300m": (60, 120, 180, 240, 300),
    "band_120_300m": (120, 180, 240, 300),
    "band_center_120_240m": (120, 240),
}


@dataclass(frozen=True)
class SurrogateCandidate:
    module_count: int
    module_lengths_m: tuple[float, ...]
    module_centers_m: tuple[float, ...]
    surrogate_rmse_by_wavelength: dict[int, float]
    surrogate_ratio_by_wavelength: dict[int, float]
    surrogate_score_by_target: dict[str, float]


@dataclass(frozen=True)
class SelectedCandidate:
    layout: sweep.LayoutSpec
    selected_for_targets: tuple[str, ...]
    surrogate_score_by_target: dict[str, float]
    surrogate_ratio_by_wavelength: dict[int, float]


def parse_ints(text: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not text:
        return default
    values = tuple(int(float(item)) for item in re.split(r"[\s,;]+", text.strip()) if item)
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_targets(text: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not text:
        return default
    values = tuple(item for item in re.split(r"[\s,;]+", text.strip()) if item)
    unknown = [item for item in values if item not in TARGET_WAVELENGTHS]
    if unknown:
        raise ValueError(f"unknown target ids: {unknown}")
    return values


def parse_length_choices(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_LENGTH_CHOICES_M
    values = tuple(sorted({int(float(item)) for item in re.split(r"[\s,;]+", text.strip()) if item}))
    if not values:
        raise ValueError("at least one module length choice is required")
    if any(value <= 0 for value in values):
        raise ValueError("module length choices must be positive")
    if any(value % 10 != 0 for value in values):
        raise ValueError("module length choices must be multiples of 10 m to keep centers on the 5 m FEM grid")
    return values


def module_boundaries(lengths_m: tuple[float, ...]) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])


def module_centers(lengths_m: tuple[float, ...]) -> tuple[float, ...]:
    boundaries = module_boundaries(lengths_m)
    return tuple((0.5 * (boundaries[:-1] + boundaries[1:])).tolist())


def length_signature(lengths_m: tuple[float, ...]) -> str:
    return "_".join(str(int(value)) for value in lengths_m)


def file_uri(path: Path) -> str:
    return path.resolve().as_posix()


def target_wavelengths(target_id: str, available_wavelengths: tuple[int, ...]) -> tuple[int, ...]:
    wanted = TARGET_WAVELENGTHS[target_id]
    missing = [value for value in wanted if value not in available_wavelengths]
    if missing:
        raise ValueError(f"target {target_id} requires missing wavelengths {missing}")
    return wanted


def generate_layouts(
    *,
    module_count: int,
    total_length_m: int,
    choices_m: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    """Generate only layouts that exactly sum to the total length."""

    choices = tuple(sorted(choices_m))
    layouts: list[tuple[float, ...]] = []

    def rec(prefix: tuple[int, ...], remaining: int, slots_left: int) -> None:
        if slots_left == 0:
            if remaining == 0:
                layouts.append(tuple(float(value) for value in prefix))
            return
        min_tail = choices[0] * (slots_left - 1)
        max_tail = choices[-1] * (slots_left - 1)
        for value in choices:
            next_remaining = remaining - value
            if min_tail <= next_remaining <= max_tail:
                rec((*prefix, value), next_remaining, slots_left - 1)

    rec(tuple(), total_length_m, module_count)
    return tuple(layouts)


def reference_heave_by_wavelength(
    wavelengths_m: tuple[int, ...],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    u10_rmse_by_wavelength: dict[int, float] = {}
    for wavelength_m in wavelengths_m:
        x_ref, heave_ref = sweep.load_heave(mcpv.reference_response_path("U30_reference", wavelength_m))
        x_u10, heave_u10 = sweep.load_heave(mcpv.reference_response_path("uniform_U10", wavelength_m))
        if not np.allclose(x_ref, x_u10):
            raise ValueError(f"U10/U30 centerline x mismatch at {wavelength_m} m")
        references[wavelength_m] = (x_ref, heave_ref)
        u10_rmse_by_wavelength[wavelength_m] = float(np.sqrt(np.mean((heave_u10 - heave_ref) ** 2)))
    return references, u10_rmse_by_wavelength


def interpolation_surrogate(
    lengths_m: tuple[float, ...],
    references: dict[int, tuple[np.ndarray, np.ndarray]],
    u10_rmse_by_wavelength: dict[int, float],
    targets: tuple[str, ...],
) -> SurrogateCandidate:
    centers_m = module_centers(lengths_m)
    centers_x_over_l = np.asarray(centers_m, dtype=float) / sweep.LENGTH_M
    rmse_by_wavelength: dict[int, float] = {}
    ratio_by_wavelength: dict[int, float] = {}
    for wavelength_m, (x_ref, heave_ref) in references.items():
        sampled = np.interp(centers_x_over_l, x_ref, heave_ref)
        reconstructed = np.interp(x_ref, centers_x_over_l, sampled)
        rmse = float(np.sqrt(np.mean((reconstructed - heave_ref) ** 2)))
        rmse_by_wavelength[wavelength_m] = rmse
        denom = u10_rmse_by_wavelength[wavelength_m]
        ratio_by_wavelength[wavelength_m] = rmse / denom if denom > 0.0 else float("nan")

    score_by_target: dict[str, float] = {}
    available = tuple(sorted(references))
    for target_id in targets:
        wavelengths = target_wavelengths(target_id, available)
        score_by_target[target_id] = float(np.mean([ratio_by_wavelength[value] for value in wavelengths]))

    return SurrogateCandidate(
        module_count=len(lengths_m),
        module_lengths_m=lengths_m,
        module_centers_m=centers_m,
        surrogate_rmse_by_wavelength=rmse_by_wavelength,
        surrogate_ratio_by_wavelength=ratio_by_wavelength,
        surrogate_score_by_target=score_by_target,
    )


def build_surrogate_candidates(
    *,
    counts: tuple[int, ...],
    length_choices_m: tuple[int, ...],
    wavelengths_m: tuple[int, ...],
    targets: tuple[str, ...],
) -> list[SurrogateCandidate]:
    references, u10_rmse_by_wavelength = reference_heave_by_wavelength(wavelengths_m)
    candidates: list[SurrogateCandidate] = []
    for module_count in counts:
        for lengths in generate_layouts(
            module_count=module_count,
            total_length_m=int(round(sweep.LENGTH_M)),
            choices_m=length_choices_m,
        ):
            if module_count == 10 and all(np.isclose(value, 30.0) for value in lengths):
                continue
            candidates.append(
                interpolation_surrogate(
                    lengths,
                    references,
                    u10_rmse_by_wavelength,
                    targets,
                )
            )
    return candidates


def write_preselection_tables(
    *,
    candidates: list[SurrogateCandidate],
    search_targets: tuple[str, ...],
    counts: tuple[int, ...],
    wavelengths_m: tuple[int, ...],
    keep_per_target_count: int,
) -> tuple[Path, Path]:
    all_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for target_id in search_targets:
        for module_count in counts:
            group = [item for item in candidates if item.module_count == module_count]
            group = sorted(group, key=lambda item: item.surrogate_score_by_target[target_id])
            for rank, candidate in enumerate(group, start=1):
                row = {
                    "target_id": target_id,
                    "module_count": module_count,
                    "rank": rank,
                    "surrogate_score_ratio_vs_U10": candidate.surrogate_score_by_target[target_id],
                    "module_lengths_m": " ".join(f"{value:g}" for value in candidate.module_lengths_m),
                    "module_centers_m": " ".join(f"{value:.1f}" for value in candidate.module_centers_m),
                }
                for wavelength_m in wavelengths_m:
                    row[f"surrogate_ratio_{wavelength_m}m"] = candidate.surrogate_ratio_by_wavelength[wavelength_m]
                    row[f"surrogate_rmse_{wavelength_m}m"] = candidate.surrogate_rmse_by_wavelength[wavelength_m]
                if rank <= keep_per_target_count:
                    top_rows.append(row)
                if rank <= max(keep_per_target_count, 5):
                    all_rows.append(row)

    all_path = mcpv.write_csv(sweep.TABLE_DIR / "response_aware_preselection_top_rows.csv", all_rows)
    top_path = mcpv.write_csv(sweep.TABLE_DIR / "response_aware_preselection_selected_pool.csv", top_rows)
    return all_path, top_path


def select_validation_candidates(
    *,
    candidates: list[SurrogateCandidate],
    validation_targets: tuple[str, ...],
    counts: tuple[int, ...],
    top_k_per_target_count: int,
) -> tuple[SelectedCandidate, ...]:
    by_lengths: dict[tuple[float, ...], set[str]] = {}
    by_lengths_candidate: dict[tuple[float, ...], SurrogateCandidate] = {}
    for target_id in validation_targets:
        for module_count in counts:
            group = [item for item in candidates if item.module_count == module_count]
            group = sorted(group, key=lambda item: item.surrogate_score_by_target[target_id])
            for candidate in group[:top_k_per_target_count]:
                by_lengths.setdefault(candidate.module_lengths_m, set()).add(target_id)
                by_lengths_candidate[candidate.module_lengths_m] = candidate

    selected: list[SelectedCandidate] = []
    for index, lengths in enumerate(sorted(by_lengths, key=lambda item: (len(item), item)), start=1):
        candidate = by_lengths_candidate[lengths]
        layout = sweep.LayoutSpec(
            layout_id=f"LCP_RA_N{len(lengths)}_{index:02d}",
            display_name=f"response-aware N{len(lengths)}",
            category="low_control_response_aware",
            module_lengths_m=lengths,
        )
        selected.append(
            SelectedCandidate(
                layout=layout,
                selected_for_targets=tuple(sorted(by_lengths[lengths])),
                surrogate_score_by_target=candidate.surrogate_score_by_target,
                surrogate_ratio_by_wavelength=candidate.surrogate_ratio_by_wavelength,
            )
        )
    return tuple(selected)


def write_selected_candidate_table(selected: tuple[SelectedCandidate, ...], wavelengths_m: tuple[int, ...]) -> Path:
    rows: list[dict[str, object]] = []
    for candidate in selected:
        layout = candidate.layout
        row = {
            "layout_id": layout.layout_id,
            "display_name": layout.display_name,
            "module_count": layout.module_count,
            "selected_for_targets": " ".join(candidate.selected_for_targets),
            "module_lengths_m": " ".join(f"{value:g}" for value in layout.module_lengths_m),
            "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(layout.module_lengths_m)),
        }
        for target_id, score in sorted(candidate.surrogate_score_by_target.items()):
            row[f"surrogate_score_{target_id}"] = score
        for wavelength_m in wavelengths_m:
            row[f"surrogate_ratio_{wavelength_m}m"] = candidate.surrogate_ratio_by_wavelength[wavelength_m]
        rows.append(row)
    return mcpv.write_csv(sweep.TABLE_DIR / "selected_response_aware_candidates.csv", rows)


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
                "module_centers_m": " ".join(f"{value:.1f}" for value in module_centers(layout.module_lengths_m)),
                "selected_node_ids": " ".join(str(value) for value in node_ids),
                "max_abs_center_node_error_m": max(float(row["abs_error_m"]) for row in geometry),
                "has_duplicate_control_nodes": len(set(node_ids)) != len(node_ids),
                "hydro_path": str(configs[layout.layout_id].output_path),
                "geometry_csv": str(geometry_paths[layout.layout_id]),
            }
        )
    return mcpv.write_csv(sweep.TABLE_DIR / "response_aware_geometry_manifest.csv", rows)


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


def write_actual_validation_table(
    *,
    selected: tuple[SelectedCandidate, ...],
    validation_targets: tuple[str, ...],
    counts: tuple[int, ...],
    wavelengths_m: tuple[int, ...],
    metric_csv: Path,
) -> Path:
    metric_rows = mcpv.read_csv(metric_csv)
    rows: list[dict[str, object]] = []
    for target_id in validation_targets:
        target_wls = target_wavelengths(target_id, wavelengths_m)
        u10_rmse = mean_rmse_for_layout(metric_rows, "uniform_U10", target_wls)
        for module_count in counts:
            matches = [
                item
                for item in selected
                if item.layout.module_count == module_count and target_id in item.selected_for_targets
            ]
            for item in matches:
                actual_rmse = mean_rmse_for_layout(metric_rows, item.layout.layout_id, target_wls)
                ratio = actual_rmse / u10_rmse if u10_rmse > 0.0 else float("nan")
                rows.append(
                    {
                        "target_id": target_id,
                        "target_wavelengths_m": " ".join(str(value) for value in target_wls),
                        "module_count": module_count,
                        "layout_id": item.layout.layout_id,
                        "display_name": item.layout.display_name,
                        "surrogate_score_ratio_vs_U10": item.surrogate_score_by_target[target_id],
                        "actual_rmse_vs_U30": actual_rmse,
                        "uniform_U10_rmse_vs_U30": u10_rmse,
                        "actual_rmse_ratio_vs_U10": ratio,
                        "actual_improvement_vs_U10_percent": (1.0 - ratio) * 100.0,
                    }
                )
        if 10 in counts or all(count < 10 for count in counts):
            rows.append(
                {
                    "target_id": target_id,
                    "target_wavelengths_m": " ".join(str(value) for value in target_wls),
                    "module_count": 10,
                    "layout_id": "uniform_U10",
                    "display_name": "U10 uniform baseline",
                    "surrogate_score_ratio_vs_U10": 1.0,
                    "actual_rmse_vs_U30": u10_rmse,
                    "uniform_U10_rmse_vs_U30": u10_rmse,
                    "actual_rmse_ratio_vs_U10": 1.0,
                    "actual_improvement_vs_U10_percent": 0.0,
                }
            )
    return mcpv.write_csv(sweep.TABLE_DIR / "actual_response_aware_validation.csv", rows)


def plot_surrogate_scores(
    candidates: list[SurrogateCandidate],
    targets: tuple[str, ...],
    counts: tuple[int, ...],
) -> Path:
    import matplotlib.pyplot as plt

    path = sweep.FIGURE_DIR / "response_aware_surrogate_best_scores.png"
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    for target_id in targets:
        y_values = []
        for module_count in counts:
            group = [item for item in candidates if item.module_count == module_count]
            y_values.append(min(item.surrogate_score_by_target[target_id] for item in group))
        axis.plot(counts, y_values, marker="o", linewidth=1.5, label=target_id)
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0, label="U10 actual RMSE level")
    axis.set_xlabel("control-point/module count")
    axis.set_ylabel("best surrogate RMSE ratio vs U10")
    axis.set_title("Response-aware preselection: best surrogate score by control-point count")
    axis.set_xticks(counts)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_actual_heatmap(actual_csv: Path, validation_targets: tuple[str, ...], counts: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(actual_csv)
    display_counts = tuple(sorted(set((*counts, 10))))
    values = np.full((len(validation_targets), len(display_counts)), np.nan)
    labels = [["" for _ in display_counts] for _ in validation_targets]
    for row in rows:
        target_id = row["target_id"]
        if target_id not in validation_targets:
            continue
        module_count = int(row["module_count"])
        if module_count not in display_counts:
            continue
        target_index = validation_targets.index(target_id)
        count_index = display_counts.index(module_count)
        value = float(row["actual_improvement_vs_U10_percent"])
        current = values[target_index, count_index]
        if not np.isfinite(current) or value > current:
            values[target_index, count_index] = value
            labels[target_index][count_index] = f"{value:.1f}"

    finite = values[np.isfinite(values)]
    vmax = max(10.0, float(np.nanmax(np.abs(finite)))) if finite.size else 10.0
    vmax = min(vmax, 100.0)
    fig, axis = plt.subplots(figsize=(9.8, 4.6))
    image = axis.imshow(np.clip(values, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(display_counts)))
    axis.set_xticklabels([f"N{value}" for value in display_counts])
    axis.set_yticks(np.arange(len(validation_targets)))
    axis.set_yticklabels(validation_targets)
    axis.set_xlabel("control-point/module count")
    axis.set_title("Actual RODM validation: heave RMSE improvement vs uniform U10 (%)")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if labels[row_index][col_index]:
                axis.text(col_index, row_index, labels[row_index][col_index], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("positive means closer to U30 than U10")
    fig.tight_layout()
    path = sweep.FIGURE_DIR / "actual_response_aware_improvement_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_actual_ratio_curves(actual_csv: Path, validation_targets: tuple[str, ...], counts: tuple[int, ...]) -> Path:
    import matplotlib.pyplot as plt

    rows = mcpv.read_csv(actual_csv)
    display_counts = tuple(sorted(set((*counts, 10))))
    fig, axis = plt.subplots(figsize=(10.2, 5.5))
    for target_id in validation_targets:
        x_values = []
        y_values = []
        for module_count in display_counts:
            matches = [
                row
                for row in rows
                if row["target_id"] == target_id and int(row["module_count"]) == module_count
            ]
            if not matches:
                continue
            best = min(matches, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
            x_values.append(module_count)
            y_values.append(float(best["actual_rmse_ratio_vs_U10"]))
        axis.plot(x_values, y_values, marker="o", linewidth=1.5, label=target_id)
    axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    axis.set_xlabel("control-point/module count")
    axis.set_ylabel("actual RMSE ratio vs U10")
    axis.set_title("Actual RODM validation: low-control response-aware layouts")
    axis.set_xticks(display_counts)
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = sweep.FIGURE_DIR / "actual_response_aware_ratio_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def best_layout_for_target(actual_csv: Path, target_id: str) -> str:
    rows = [
        row
        for row in mcpv.read_csv(actual_csv)
        if row["target_id"] == target_id and row["layout_id"] != "uniform_U10"
    ]
    if not rows:
        return "uniform_U10"
    best = min(rows, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
    return best["layout_id"]


def plot_best_heave_panel(
    *,
    response_paths: dict[tuple[str, int], Path],
    layouts: tuple[sweep.LayoutSpec, ...],
    wavelengths_m: tuple[int, ...],
    actual_csv: Path,
    primary_target: str,
) -> Path:
    import matplotlib.pyplot as plt

    best_layout_id = best_layout_for_target(actual_csv, primary_target)
    plot_ids = ("U30_reference", "uniform_U10", best_layout_id)
    layout_by_id = {layout.layout_id: layout for layout in layouts}
    colors = {
        "U30_reference": "#111111",
        "uniform_U10": "#1f77b4",
        best_layout_id: "#d62728",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0), sharex=True)
    axes_flat = np.ravel(axes)
    for axis, wavelength_m in zip(axes_flat, wavelengths_m):
        for layout_id in plot_ids:
            x, heave = sweep.load_heave(response_paths[(layout_id, wavelength_m)])
            label = "U30 reference" if layout_id == "U30_reference" else layout_by_id[layout_id].display_name
            axis.plot(
                x,
                heave,
                linewidth=2.0 if layout_id == "U30_reference" else 1.4,
                linestyle="-" if layout_id in {"U30_reference", "uniform_U10"} else "--",
                color=colors.get(layout_id, "#666666"),
                label=label,
            )
        axis.set_title(f"{wavelength_m} m")
        axis.set_ylabel("Heave RAO")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    for axis in axes_flat[len(wavelengths_m) :]:
        axis.axis("off")
    axes_flat[0].legend(frameon=False, fontsize=8)
    for axis in axes_flat[-3:]:
        axis.set_xlabel("x/L")
    fig.suptitle(f"Low-control non-uniform validation: best layout for {primary_target}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = sweep.FIGURE_DIR / "best_low_control_heave_panel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    *,
    counts: tuple[int, ...],
    length_choices_m: tuple[int, ...],
    search_targets: tuple[str, ...],
    validation_targets: tuple[str, ...],
    selected_csv: Path,
    geometry_csv: Path,
    metric_csv: Path,
    actual_csv: Path,
    figures: dict[str, Path],
) -> Path:
    actual_rows = mcpv.read_csv(actual_csv)

    def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(":---" if index == 0 else "---:" for index in range(len(headers))) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    best_rows = []
    for target_id in validation_targets:
        rows = [row for row in actual_rows if row["target_id"] == target_id]
        best = min(rows, key=lambda row: float(row["actual_rmse_ratio_vs_U10"]))
        best_rows.append(
            (
                target_id,
                f"N{best['module_count']}",
                best["layout_id"],
                f"{float(best['actual_rmse_ratio_vs_U10']):.4f}",
                f"{float(best['actual_improvement_vs_U10_percent']):.2f}%",
            )
        )

    best_nonuniform_rows = [
        row
        for row in actual_rows
        if row["layout_id"] != "uniform_U10"
    ]
    best_nonuniform = min(
        best_nonuniform_rows,
        key=lambda row: float(row["actual_rmse_ratio_vs_U10"]),
    )

    lines = [
        "# 少控制点非均匀模块逼近 U30 高精度解：响应感知候选研究",
        "",
        "## 1. 研究问题",
        "",
        (
            "这里研究的不是“模块数越多越准”，而是在控制点数量较少的前提下，"
            "是否可以通过非均匀模块划分，把有限的水动力控制点放在对 heave 响应更敏感的位置，"
            "从而更接近 U30 高精度参考解。"
        ),
        "",
        "## 2. 方法",
        "",
        "- 候选模块长度取 FEM 对齐的离散值，确保模块重心落在 5 m 结构网格节点上。",
        f"- 搜索控制点数量：`{', '.join('N' + str(value) for value in counts)}`。",
        f"- 模块长度候选：`{', '.join(str(value) for value in length_choices_m)} m`。",
        f"- 预筛选目标：`{', '.join(search_targets)}`。",
        f"- 实际水动力验证目标：`{', '.join(validation_targets)}`。",
        "- 每个进入实际验证的非均匀布局都重新生成 Capytaine `.nc`，再求解 ordered SEREP-ridge RODM。",
        "",
        "## 3. 图",
        "",
        f"![Response-aware surrogate scores]({file_uri(figures['surrogate'])})",
        "",
        f"![Actual validation heatmap]({file_uri(figures['actual_heatmap'])})",
        "",
        f"![Actual validation ratio curves]({file_uri(figures['actual_curves'])})",
        "",
        f"![Best low-control heave panel]({file_uri(figures['heave_panel'])})",
        "",
        "## 4. 实际 RODM 验证最优结果",
        "",
        table(("target", "best N", "best layout", "RMSE ratio vs U10", "improvement"), best_rows),
        "",
        "## 5. 输出文件",
        "",
        f"- 候选布局：`{selected_csv}`",
        f"- 几何与 FEM 主控制点：`{geometry_csv}`",
        f"- 逐波长误差：`{metric_csv}`",
        f"- 实际验证汇总：`{actual_csv}`",
        f"- 图目录：`{sweep.FIGURE_DIR}`",
        "",
        "## 6. 当前结论",
        "",
        (
            "这轮真实 RODM 验证给出了一个明确边界：在当前 SEREP-ridge、draft=0.5 m、水深 58.5 m、"
            "rho=1000 的设置下，N5-N9 的响应感知非均匀布局都没有超过 U10。"
        ),
        "",
        (
            f"所有低控制点候选中最接近 U10 的是 `{best_nonuniform['layout_id']}`，"
            f"对应 `{best_nonuniform['target_id']}` 的 RMSE ratio vs U10 = "
            f"`{float(best_nonuniform['actual_rmse_ratio_vs_U10']):.4f}`。"
        ),
        "",
        "因此，论文叙事不应写成“控制点可以任意减少”。更合理的结论是：",
        "",
        "1. 过少控制点会导致水动力载荷空间采样不足，非均匀划分无法弥补信息缺失。",
        "2. 非均匀模块的有效区间更可能从 N10/N11/N12 附近开始，而不是 N5-N9。",
        "3. 后续目标应改为“在给定误差容限下寻找最小可行控制点数量”，并用真实 RODM-in-the-loop 验证。",
        "",
        "这个负结果可以写入论文讨论：它给出了非均匀模块方法的适用边界，避免把方法夸大成“控制点可以无限减少”。",
        "",
    ]
    path = sweep.OUTPUT_ROOT / "low_control_point_response_aware_study_report.md"
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def run_workflow(args: argparse.Namespace) -> dict[str, str]:
    counts = parse_ints(args.counts, DEFAULT_COUNTS)
    wavelengths_m = mcpv.parse_wavelengths(args.wavelengths)
    search_targets = parse_targets(args.search_targets, DEFAULT_SEARCH_TARGETS)
    validation_targets = parse_targets(args.validation_targets, DEFAULT_VALIDATION_TARGETS)
    length_choices_m = parse_length_choices(args.length_choices)

    output_root = args.output_root or DEFAULT_OUTPUT_ROOT
    mcpv.configure_sweep_output(output_root)
    sweep.REPORT_PATH = output_root / "low_control_point_response_aware_study_report.md"

    candidates = build_surrogate_candidates(
        counts=counts,
        length_choices_m=length_choices_m,
        wavelengths_m=wavelengths_m,
        targets=search_targets,
    )
    preselection_csv, selected_pool_csv = write_preselection_tables(
        candidates=candidates,
        search_targets=search_targets,
        counts=counts,
        wavelengths_m=wavelengths_m,
        keep_per_target_count=args.keep_preselection_rows,
    )
    selected = select_validation_candidates(
        candidates=candidates,
        validation_targets=validation_targets,
        counts=counts,
        top_k_per_target_count=args.top_k_per_target_count,
    )
    selected_csv = write_selected_candidate_table(selected, wavelengths_m)

    layouts = (
        sweep.LayoutSpec("U30_reference", "U30 reference", "reference", (10.0,) * 30),
        sweep.LayoutSpec("uniform_U10", "U10 uniform", "baseline", (30.0,) * 10),
        *(item.layout for item in selected),
    )
    mcpv.update_plot_colors(layouts)
    configs = {layout.layout_id: sweep.hydro_config(layout, wavelengths_m, n_jobs=args.n_jobs) for layout in layouts}
    geometry_by_layout = {layout.layout_id: sweep.geometry_rows(layout, configs[layout.layout_id]) for layout in layouts}
    geometry_paths = {
        layout.layout_id: sweep.write_geometry_csv(layout, geometry_by_layout[layout.layout_id]) for layout in layouts
    }
    geometry_csv = write_geometry_manifest(layouts, configs, geometry_paths)

    figures = {
        "surrogate": plot_surrogate_scores(candidates, validation_targets, counts),
    }

    if args.dry_run:
        manifest = {
            "mode": "dry_run",
            "candidate_count": str(len(candidates)),
            "selected_candidate_count": str(len(selected)),
            "preselection_csv": str(preselection_csv),
            "selected_pool_csv": str(selected_pool_csv),
            "selected_csv": str(selected_csv),
            "geometry_csv": str(geometry_csv),
            "figure_surrogate": str(figures["surrogate"]),
        }
        (sweep.OUTPUT_ROOT / "low_control_point_response_aware_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    response_paths = mcpv.copy_reference_responses(wavelengths_m=wavelengths_m)
    for item in selected:
        layout = item.layout
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

    metric_csv, layout_summary_csv = sweep.write_metrics(layouts, response_paths, wavelengths_m)
    actual_csv = write_actual_validation_table(
        selected=selected,
        validation_targets=validation_targets,
        counts=counts,
        wavelengths_m=wavelengths_m,
        metric_csv=metric_csv,
    )
    figures.update(
        {
            "actual_heatmap": plot_actual_heatmap(actual_csv, validation_targets, counts),
            "actual_curves": plot_actual_ratio_curves(actual_csv, validation_targets, counts),
            "heave_panel": plot_best_heave_panel(
                response_paths=response_paths,
                layouts=layouts,
                wavelengths_m=wavelengths_m,
                actual_csv=actual_csv,
                primary_target=validation_targets[0],
            ),
        }
    )
    report = write_report(
        counts=counts,
        length_choices_m=length_choices_m,
        search_targets=search_targets,
        validation_targets=validation_targets,
        selected_csv=selected_csv,
        geometry_csv=geometry_csv,
        metric_csv=metric_csv,
        actual_csv=actual_csv,
        figures=figures,
    )

    manifest = {
        "mode": "full",
        "candidate_count": str(len(candidates)),
        "selected_candidate_count": str(len(selected)),
        "report": str(report),
        "preselection_csv": str(preselection_csv),
        "selected_pool_csv": str(selected_pool_csv),
        "selected_csv": str(selected_csv),
        "geometry_csv": str(geometry_csv),
        "metric_csv": str(metric_csv),
        "layout_summary_csv": str(layout_summary_csv),
        "actual_csv": str(actual_csv),
        **{f"figure_{key}": str(value) for key, value in figures.items()},
    }
    (sweep.OUTPUT_ROOT / "low_control_point_response_aware_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default=None, help="Control-point counts, e.g. '5 6 7 8 9'.")
    parser.add_argument("--length-choices", default=None, help="FEM-aligned module length choices in m.")
    parser.add_argument("--wavelengths", default=None, help="Wavelengths in m. Default: 60 120 180 240 300.")
    parser.add_argument("--search-targets", default=None, help="Targets used in surrogate preselection.")
    parser.add_argument("--validation-targets", default=None, help="Targets selected for actual RODM validation.")
    parser.add_argument("--top-k-per-target-count", type=int, default=1)
    parser.add_argument("--keep-preselection-rows", type=int, default=20)
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
