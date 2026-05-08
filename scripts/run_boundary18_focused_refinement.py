"""Focused 18-variable refinement for the 10x10 hinge-stiffness paper result.

The first full-range DOE established that the most informative transition
region sits roughly between ``1e7`` and ``1e9``.  This script densifies that
region with manufacturable 18D patterns:

* dense uniform baselines;
* x/y orientation-asymmetric designs;
* smooth center/edge/gradient profiles;
* a small set of localized boundary changes;
* smooth random log-space profiles.

It then combines the focused rows with the existing full-range and low-stiffness
summaries, marks a global three-objective Pareto set, and writes figures that
directly compare non-uniform candidates against nearby uniform baselines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.optimization import (  # noqa: E402
    BoundaryStiffnessDesign,
    MetricObjective,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    evaluate_design_response,
    mark_pareto_rows,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
    solve_complex_hinge_case,
)

from run_boundary18_fullrange_single_frequency import (  # noqa: E402
    summarize_connector_relative_motion,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "boundary18_focused_refinement"
FULLRANGE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "boundary18_fullrange_single_frequency"
    / "boundary18_fullrange_summary.csv"
)
LOW_STIFFNESS_SUMMARY = (
    REPO_ROOT
    / "results"
    / "low_stiffness_sensitivity"
    / "low_stiffness_sensitivity_summary.csv"
)


@dataclass(frozen=True)
class FocusedSample:
    """One manufacturable 18D stiffness pattern."""

    name: str
    values: tuple[float, ...]
    family: str
    description: str

    def __post_init__(self) -> None:
        if len(self.values) != 18:
            raise ValueError(f"{self.name}: expected 18 values, got {len(self.values)}")
        if any(value < 0.0 for value in self.values):
            raise ValueError(f"{self.name}: stiffness values must be non-negative")


def _safe_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{float(value):.2e}".replace("+", "").replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _uniform(value: float) -> tuple[float, ...]:
    return tuple([float(value)] * 18)


def _center_profile(low: float, high: float) -> tuple[float, ...]:
    center = 5
    max_distance = 4
    values = []
    for boundary in range(1, 10):
        weight = 1.0 - abs(boundary - center) / max_distance
        values.append(low * (high / low) ** weight)
    return tuple(float(value) for value in values)


def _edge_profile(low: float, high: float) -> tuple[float, ...]:
    center = 5
    max_distance = 4
    values = []
    for boundary in range(1, 10):
        weight = abs(boundary - center) / max_distance
        values.append(low * (high / low) ** weight)
    return tuple(float(value) for value in values)


def _gradient_profile(low: float, high: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.geomspace(low, high, 9))


def _smooth_random_profile(rng: np.random.Generator, low: float, high: float) -> tuple[float, ...]:
    raw = rng.uniform(np.log10(low), np.log10(high), size=9)
    padded = np.pad(raw, (1, 1), mode="edge")
    smooth = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    return tuple(float(10.0**value) for value in smooth)


def _add_sample(samples: dict[str, FocusedSample], sample: FocusedSample) -> None:
    """Append a sample unless the same label has already been used."""

    samples.setdefault(sample.name, sample)


def generate_focused_samples(*, random_count: int = 28, seed: int = 20260508) -> tuple[FocusedSample, ...]:
    """Return focused 18D samples in the transition interval ``1e7``-``1e9``."""

    samples: dict[str, FocusedSample] = {}
    uniform_levels = (
        1.0e7,
        1.7782794100389228e7,
        3.162277660168379e7,
        5.623413251903491e7,
        1.0e8,
        1.7782794100389228e8,
        3.1622776601683795e8,
        5.623413251903491e8,
        1.0e9,
    )
    for value in uniform_levels:
        _add_sample(
            samples,
            FocusedSample(
                f"uniform_{_safe_label(value)}",
                _uniform(value),
                "uniform_dense",
                "Dense uniform baseline in the transition stiffness interval.",
            ),
        )

    orientation_levels = (1.0e7, 3.162277660168379e7, 1.0e8, 3.1622776601683795e8, 1.0e9)
    for x_value in orientation_levels:
        for y_value in orientation_levels:
            if np.isclose(x_value, y_value):
                continue
            _add_sample(
                samples,
                FocusedSample(
                    f"orient_x_{_safe_label(x_value)}_y_{_safe_label(y_value)}",
                    tuple([x_value] * 9 + [y_value] * 9),
                    "orientation_grid",
                    "x and y complete boundaries use different uniform stiffness levels.",
                ),
            )

    profile_bounds = (
        (1.0e7, 1.0e8),
        (1.0e7, 3.1622776601683795e8),
        (1.0e7, 1.0e9),
        (3.162277660168379e7, 3.1622776601683795e8),
        (3.162277660168379e7, 1.0e9),
        (1.0e8, 1.0e9),
    )
    for low, high in profile_bounds:
        center = _center_profile(low, high)
        edge = _edge_profile(low, high)
        gradient = _gradient_profile(low, high)
        reverse_gradient = tuple(reversed(gradient))
        label_low = _safe_label(low)
        label_high = _safe_label(high)
        _add_sample(
            samples,
            FocusedSample(
                f"center_{label_low}_{label_high}",
                center + center,
                "smooth_center_edge",
                "Both orientations are stiffest near the platform center.",
            ),
        )
        _add_sample(
            samples,
            FocusedSample(
                f"edge_{label_low}_{label_high}",
                edge + edge,
                "smooth_center_edge",
                "Both orientations are stiffest near outer internal boundaries.",
            ),
        )
        _add_sample(
            samples,
            FocusedSample(
                f"x_gradient_y_uniform_{label_low}_{label_high}",
                gradient + tuple([low] * 9),
                "smooth_gradient",
                "x boundaries follow a smooth gradient while y boundaries stay low.",
            ),
        )
        _add_sample(
            samples,
            FocusedSample(
                f"x_uniform_y_gradient_{label_low}_{label_high}",
                tuple([low] * 9) + gradient,
                "smooth_gradient",
                "y boundaries follow a smooth gradient while x boundaries stay low.",
            ),
        )
        _add_sample(
            samples,
            FocusedSample(
                f"opposed_gradients_{label_low}_{label_high}",
                gradient + reverse_gradient,
                "smooth_gradient",
                "x and y boundaries follow opposite smooth gradients.",
            ),
        )

    for base, high in ((1.0e7, 1.0e8), (1.0e7, 1.0e9), (1.0e8, 1.0e9)):
        for orientation, offset in (("x", 0), ("y", 9)):
            for boundary in (1, 3, 5, 7, 9):
                values = list(_uniform(base))
                values[offset + boundary - 1] = high
                _add_sample(
                    samples,
                    FocusedSample(
                        f"local_{orientation}{boundary}_{_safe_label(base)}_{_safe_label(high)}",
                        tuple(float(value) for value in values),
                        "localized_boundary",
                        "One complete internal boundary is locally stiffened relative to a uniform base.",
                    ),
                )

    rng = np.random.default_rng(seed)
    for index in range(random_count):
        x_profile = _smooth_random_profile(rng, 1.0e7, 1.0e9)
        y_profile = _smooth_random_profile(rng, 1.0e7, 1.0e9)
        _add_sample(
            samples,
            FocusedSample(
                f"smooth_random_log_{index + 1:02d}",
                x_profile + y_profile,
                "smooth_random",
                "Smooth random log-space profile in the transition stiffness interval.",
            ),
        )

    return tuple(samples.values())


def _read_rows(path: Path, *, sample_set: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            item = dict(row)
            item["source_summary"] = str(path)
            item["sample_set"] = sample_set
            rows.append(item)
    return rows


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _is_true(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _display_label(label: str) -> str:
    def number_label(token: str) -> str:
        mapping = {
            "0": "0",
            "1p00e07": "$10^7$",
            "1p78e07": "$1.78\\times10^7$",
            "3p16e07": "$3.16\\times10^7$",
            "5p62e07": "$5.62\\times10^7$",
            "1p00e08": "$10^8$",
            "1p78e08": "$1.78\\times10^8$",
            "3p16e08": "$3.16\\times10^8$",
            "5p62e08": "$5.62\\times10^8$",
            "1p00e09": "$10^9$",
            "1p00e10": "$10^{10}$",
            "1p00e11": "$10^{11}$",
        }
        return mapping.get(token, token)

    if label.startswith("uniform_"):
        suffix = label.removeprefix("uniform_")
        return "uniform\n" + number_label(suffix)
    if label.startswith("orient_x_"):
        x_token, y_token = label.removeprefix("orient_x_").split("_y_")
        return f"x {number_label(x_token)}\ny {number_label(y_token)}"
    if label.startswith("local_"):
        parts = label.split("_")
        if len(parts) >= 4:
            return f"{parts[1]}\n{number_label(parts[2])}->{number_label(parts[3])}"
    return label.replace("_", "\n")


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by design label, preferring focused rows."""

    priority = {"focused_refinement": 0, "fullrange": 1, "low_stiffness": 2}
    sorted_rows = sorted(rows, key=lambda row: priority.get(str(row.get("sample_set")), 99))
    by_label: dict[str, dict[str, Any]] = {}
    for row in sorted_rows:
        by_label.setdefault(str(row["design_label"]), row)
    return list(by_label.values())


def _global_pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective("released_rotation", "max_released_relative_rotation_envelope", minimize=True),
        MetricObjective("connector_bending", "max_connector_bending_envelope", minimize=True),
    )
    marked = mark_pareto_rows(rows, objectives)
    return marked


def _matched_uniform_gain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniform_rows = [
        row
        for row in rows
        if str(row["design_label"]).startswith("uniform_")
        and 1.0e7 <= _float(row, "boundary_stiffness_mean") <= 1.0e9
    ]
    nonuniform_pareto = [
        row
        for row in rows
        if _is_true(row.get("is_pareto", False)) and not str(row["design_label"]).startswith("uniform_")
    ]
    gain_rows: list[dict[str, Any]] = []
    for row in nonuniform_pareto:
        baseline = min(
            uniform_rows,
            key=lambda uniform: abs(_float(uniform, "mean_heave") - _float(row, "mean_heave")),
        )
        gain_rows.append(
            {
                "design_label": row["design_label"],
                "family": row.get("family", ""),
                "matched_uniform_label": baseline["design_label"],
                "mean_heave": row["mean_heave"],
                "matched_uniform_mean_heave": baseline["mean_heave"],
                "mean_heave_delta_percent": 100.0
                * (_float(row, "mean_heave") / _float(baseline, "mean_heave") - 1.0),
                "released_rotation_delta_percent": 100.0
                * (
                    _float(row, "max_released_relative_rotation_envelope")
                    / _float(baseline, "max_released_relative_rotation_envelope")
                    - 1.0
                ),
                "connector_bending_delta_percent": 100.0
                * (
                    _float(row, "max_connector_bending_envelope")
                    / _float(baseline, "max_connector_bending_envelope")
                    - 1.0
                ),
                "connector_shear_delta_percent": 100.0
                * (
                    _float(row, "max_connector_shear_envelope")
                    / _float(baseline, "max_connector_shear_envelope")
                    - 1.0
                ),
                "mean_heave_abs_delta_percent": abs(
                    100.0 * (_float(row, "mean_heave") / _float(baseline, "mean_heave") - 1.0)
                ),
            }
        )
    return sorted(
        gain_rows,
        key=lambda item: (
            abs(float(item["mean_heave_delta_percent"])),
            float(item["connector_bending_delta_percent"]),
        ),
    )


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=260)
    return path


def _plot(rows: list[dict[str, Any]], gain_rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    plt = _ensure_matplotlib()
    figure_root = output_root / "figures"
    paths: list[Path] = []

    mean_heave = np.array([_float(row, "mean_heave") for row in rows])
    rotation = np.array([_float(row, "max_released_relative_rotation_envelope") for row in rows])
    bending = np.array([_float(row, "max_connector_bending_envelope") for row in rows])
    pareto = np.array([_is_true(row.get("is_pareto", False)) for row in rows])
    uniform = np.array([str(row["design_label"]).startswith("uniform_") for row in rows])
    focused = np.array([row.get("sample_set") == "focused_refinement" for row in rows])
    uniform_order = [
        index
        for index in np.argsort([_float(row, "boundary_stiffness_mean") for row in rows])
        if uniform[index]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    for ax, y, ylabel in (
        (axes[0], rotation, "max released relative rotation (rad)"),
        (axes[1], bending / 1.0e6, "max connector bending envelope ($\\times10^6$)"),
    ):
        ax.scatter(mean_heave[~focused], y[~focused], s=32, color="#ced4da", alpha=0.42, label="previous DOE")
        ax.scatter(mean_heave[focused & ~pareto], y[focused & ~pareto], s=38, color="#adb5bd", alpha=0.58, label="focused dominated")
        ax.scatter(
            mean_heave[pareto & uniform],
            y[pareto & uniform],
            s=80,
            color="#f08c00",
            edgecolors="#212529",
            linewidths=0.45,
            label="uniform Pareto",
        )
        ax.scatter(
            mean_heave[pareto & ~uniform],
            y[pareto & ~uniform],
            s=62,
            color="#1c7ed6",
            marker="D",
            edgecolors="#212529",
            linewidths=0.45,
            label="non-uniform Pareto",
        )
        ax.plot(mean_heave[uniform_order], y[uniform_order], color="#495057", linewidth=1.25, label="uniform path")
        ax.set_xlabel("mean heave amplitude (m)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
    axes[0].set_title("Motion vs relative rotation")
    axes[1].set_title("Motion vs connector bending")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Focused 18D refinement in the $10^7$-$10^9$ transition interval")
    paths.append(_save(fig, figure_root / "focused18_pareto_projection.pdf"))
    plt.close(fig)

    family_names = sorted({str(row.get("family", "previous")) for row in rows if row.get("sample_set") == "focused_refinement"})
    family_counts = []
    pareto_counts = []
    for family in family_names:
        family_rows = [row for row in rows if row.get("sample_set") == "focused_refinement" and row.get("family") == family]
        family_counts.append(len(family_rows))
        pareto_counts.append(sum(1 for row in family_rows if _is_true(row.get("is_pareto", False))))
    x = np.arange(len(family_names))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.bar(x - 0.18, family_counts, width=0.36, color="#868e96", label="samples")
    ax.bar(x + 0.18, pareto_counts, width=0.36, color="#1c7ed6", label="global Pareto")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in family_names], fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Which focused sample families enter the global Pareto set?")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    paths.append(_save(fig, figure_root / "focused18_family_pareto_counts.pdf"))
    plt.close(fig)

    candidate_gain_rows = [
        row
        for row in gain_rows
        if float(row["mean_heave_abs_delta_percent"]) <= 2.0
    ]
    balanced_gain_rows = [
        row
        for row in candidate_gain_rows
        if float(row["released_rotation_delta_percent"]) <= 0.0
        and float(row["connector_bending_delta_percent"]) <= 0.0
    ]
    top_gain_rows = sorted(
        balanced_gain_rows,
        key=lambda row: (
            float(row["mean_heave_abs_delta_percent"]),
            float(row["connector_bending_delta_percent"]),
        ),
    )[:8]
    if len(top_gain_rows) < 6:
        fallback = sorted(candidate_gain_rows, key=lambda row: float(row["connector_bending_delta_percent"]))
        for row in fallback:
            if row not in top_gain_rows:
                top_gain_rows.append(row)
            if len(top_gain_rows) >= 8:
                break
    if top_gain_rows:
        labels = [_display_label(str(row["design_label"])) for row in top_gain_rows]
        x = np.arange(len(top_gain_rows))
        width = 0.24
        fig, ax = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
        ax.axhline(0.0, color="#212529", linewidth=0.8)
        ax.bar(
            x - width,
            [float(row["mean_heave_delta_percent"]) for row in top_gain_rows],
            width=width,
            color="#2f9e44",
            label="mean heave",
        )
        ax.bar(
            x,
            [float(row["released_rotation_delta_percent"]) for row in top_gain_rows],
            width=width,
            color="#5f3dc4",
            label="released rotation",
        )
        ax.bar(
            x + width,
            [float(row["connector_bending_delta_percent"]) for row in top_gain_rows],
            width=width,
            color="#e67700",
            label="connector bending",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("change vs nearest uniform baseline (%)")
        ax.set_title("Balanced non-uniform Pareto candidates vs nearest uniform designs")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        paths.append(_save(fig, figure_root / "focused18_nonuniform_gain_vs_uniform.pdf"))
        plt.close(fig)

    selected_labels = []
    for label in ("uniform_1p00e07", "uniform_1p00e08", "uniform_3p16e08", "uniform_1p00e09"):
        if any(row["design_label"] == label for row in rows):
            selected_labels.append(label)
    for row in top_gain_rows[:4]:
        if row["design_label"] not in selected_labels:
            selected_labels.append(str(row["design_label"]))

    group_names = [f"x_boundary_{index:02d}" for index in range(1, 10)] + [
        f"y_boundary_{index:02d}" for index in range(1, 10)
    ]
    rows_by_label = {str(row["design_label"]): row for row in rows}
    matrix = []
    ylabels = []
    for label in selected_labels:
        row = rows_by_label[label]
        if all(f"k_{name}" in row and str(row[f"k_{name}"]) != "" for name in group_names):
            matrix.append([np.log10(_float(row, f"k_{name}") + 1.0) for name in group_names])
            ylabels.append(_display_label(label))
    if matrix:
        fig, ax = plt.subplots(figsize=(10.4, max(4.2, 0.48 * len(matrix) + 1.8)), constrained_layout=True)
        image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="cividis", vmin=0.0, vmax=np.log10(1.0e9 + 1.0))
        ax.set_xticks(np.arange(18))
        ax.set_xticklabels([f"x{i}" for i in range(1, 10)] + [f"y{i}" for i in range(1, 10)])
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("18 complete-boundary stiffness variables")
        ax.set_title("Representative focused 18D stiffness distributions")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("log10(k + 1)")
        paths.append(_save(fig, figure_root / "focused18_representative_stiffness_profiles.pdf"))
        plt.close(fig)

    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    response_root = output_root / "responses"
    response_root.mkdir(parents=True, exist_ok=True)

    focused_summary_path = output_root / "boundary18_focused_summary.csv"
    focused_pareto_path = output_root / "boundary18_focused_pareto_summary.csv"
    design_path = output_root / "boundary18_focused_design_values.csv"

    if args.plot_only:
        if not focused_summary_path.exists():
            raise FileNotFoundError(f"Cannot plot only; missing {focused_summary_path}")
        combined_rows = []
        combined_rows.extend(_read_rows(focused_summary_path, sample_set="focused_refinement"))
        combined_rows.extend(_read_rows(FULLRANGE_SUMMARY, sample_set="fullrange"))
        combined_rows.extend(_read_rows(LOW_STIFFNESS_SUMMARY, sample_set="low_stiffness"))
        combined_rows = _deduplicate_rows(combined_rows)
        global_pareto_rows = _global_pareto_rows(combined_rows)
        gain_rows = _matched_uniform_gain_rows(global_pareto_rows)
        global_path = output_root / "boundary18_focused_global_pareto_summary.csv"
        gain_path = output_root / "boundary18_focused_nonuniform_gain_vs_uniform.csv"
        _write_csv(global_path, global_pareto_rows)
        if gain_rows:
            _write_csv(gain_path, gain_rows)
        figure_paths = _plot(global_pareto_rows, gain_rows, output_root)
        manifest = {
            "mode": "plot_only",
            "focused_sample_count": sum(1 for row in global_pareto_rows if row.get("sample_set") == "focused_refinement"),
            "global_sample_count": len(global_pareto_rows),
            "global_pareto_count": sum(1 for row in global_pareto_rows if _is_true(row["is_pareto"])),
            "focused_summary_path": str(focused_summary_path),
            "global_pareto_path": str(global_path),
            "nonuniform_gain_path": str(gain_path) if gain_rows else "",
            "figures": [str(path) for path in figure_paths],
        }
        (output_root / "boundary18_focused_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    samples = generate_focused_samples(random_count=args.random_count, seed=args.seed)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    base_case = build_complex_hinge_10x10_case(args.data_root, k_hinge=args.coupling_stiffness)
    groups = build_hinge_design_groups(base_case, "continuous_boundary")
    group_names = [group.name for group in groups]

    summary_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] solving {sample.name}", flush=True)
        design = BoundaryStiffnessDesign(
            values=sample.values,
            grouping="continuous_boundary",
            parameter="released_dof_stiffness",
            coupling_stiffness=args.coupling_stiffness,
            label=sample.name,
            meta={"family": sample.family, "sample_description": sample.description},
        )
        value_by_group = design.values_for_groups(group_names)
        case = apply_grouped_hinge_stiffness(
            base_case,
            groups,
            value_by_group,
            parameter="released_dof_stiffness",
        )
        for group in groups:
            design_rows.append(
                {
                    "design_label": sample.name,
                    "family": sample.family,
                    "sample_description": sample.description,
                    "group_name": group.name,
                    "orientation": group.orientation,
                    "released_dof_stiffness": value_by_group[group.name],
                    "hinge_line_count": len(group.hinge_indices),
                    "hinge_lines": " ".join(str(index + 1) for index in group.hinge_indices),
                }
            )

        start = time.perf_counter()
        solved = solve_complex_hinge_case(case)
        elapsed = time.perf_counter() - start
        response_path = response_root / f"response_{sample.name}.npy"
        heave_grid_path = response_root / f"heave_grid_{sample.name}.npy"
        np.save(response_path, solved.response)
        np.save(heave_grid_path, solved.heave_grid_merged)

        evaluation = evaluate_design_response(
            case,
            solved.response,
            solved.omega,
            design=design.as_dict(group_names),
            scenario={
                "omega": solved.omega,
                "frequency_index": args.frequency_index,
                "wave_direction_deg": 0.0,
                "scenario_label": "boundary18_focused_refinement",
            },
            heave_grid=solved.heave_grid_merged,
            cid_prefix=sample.name,
        )
        row = evaluation.summary_row()
        row.update(
            summarize_connector_relative_motion(
                case,
                solved.response,
                solved.omega,
                cid_prefix=f"{sample.name}_delta",
            )
        )
        row["sample_index"] = sample_index
        row["solve_elapsed_s"] = elapsed
        row["sample_set"] = "focused_refinement"
        row["source_summary"] = str(output_root / "boundary18_focused_summary.csv")
        row["response_path"] = str(response_path)
        row["heave_grid_path"] = str(heave_grid_path)
        summary_rows.append(row)

    focused_pareto = _global_pareto_rows(summary_rows)
    _write_csv(focused_summary_path, summary_rows)
    _write_csv(focused_pareto_path, focused_pareto)
    _write_csv(design_path, design_rows)

    combined_rows = []
    combined_rows.extend(_read_rows(focused_summary_path, sample_set="focused_refinement"))
    combined_rows.extend(_read_rows(FULLRANGE_SUMMARY, sample_set="fullrange"))
    combined_rows.extend(_read_rows(LOW_STIFFNESS_SUMMARY, sample_set="low_stiffness"))
    combined_rows = _deduplicate_rows(combined_rows)
    global_pareto_rows = _global_pareto_rows(combined_rows)
    gain_rows = _matched_uniform_gain_rows(global_pareto_rows)

    global_path = output_root / "boundary18_focused_global_pareto_summary.csv"
    gain_path = output_root / "boundary18_focused_nonuniform_gain_vs_uniform.csv"
    _write_csv(global_path, global_pareto_rows)
    if gain_rows:
        _write_csv(gain_path, gain_rows)

    figure_paths = _plot(global_pareto_rows, gain_rows, output_root)
    manifest = {
        "focused_sample_count": len(summary_rows),
        "focused_pareto_count": sum(1 for row in focused_pareto if _is_true(row["is_pareto"])),
        "global_sample_count": len(global_pareto_rows),
        "global_pareto_count": sum(1 for row in global_pareto_rows if _is_true(row["is_pareto"])),
        "focused_summary_path": str(focused_summary_path),
        "focused_pareto_path": str(focused_pareto_path),
        "global_pareto_path": str(global_path),
        "nonuniform_gain_path": str(gain_path) if gain_rows else "",
        "figures": [str(path) for path in figure_paths],
    }
    (output_root / "boundary18_focused_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused 18D boundary-stiffness refinement.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--coupling-stiffness", type=float, default=1.0e10)
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--random-count", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-only", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
