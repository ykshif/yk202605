"""Run a full-range 18-boundary stiffness DOE for the 10x10 hinge case.

This script is intended for the paper Section 3.3 discussion.  It keeps the
design variables in the physical interval requested for the study:

    0 <= k_i <= 1e11

The sample set is not claimed to be a global optimizer.  It is a deterministic
DOE plus Pareto screening that establishes the first non-uniform-stiffness
result set for the result-and-discussion section.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    BoundaryStiffnessDesign,
    MetricObjective,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    evaluate_design_response,
    mark_pareto_rows,
)
from offshore_energy_sim.optimization.boundary18_doe import (  # noqa: E402
    BOUNDARY18_GROUP_NAMES,
    Boundary18Sample,
)
from offshore_energy_sim.strength import (  # noqa: E402
    build_case_hinge_pair_connectors,
    harmonic_vector_norm_envelope,
    recover_connector_response,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
    solve_complex_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "boundary18_fullrange_single_frequency"


@dataclass(frozen=True)
class RepresentativeSet:
    """Selected design labels used by plots and paper tables."""

    labels: tuple[str, ...]


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


def _center_values(low_positive: float, high: float) -> tuple[float, ...]:
    """Nine-boundary profile with highest stiffness at the center boundary."""

    raw = np.array([0.0, low_positive, 1.0e8, 1.0e9, high, 1.0e9, 1.0e8, low_positive, 0.0])
    return tuple(float(value) for value in np.clip(raw, 0.0, high))


def _edge_values(low_positive: float, high: float) -> tuple[float, ...]:
    """Nine-boundary profile with highest stiffness near outer internal boundaries."""

    raw = np.array([high, 1.0e9, 1.0e8, low_positive, 0.0, low_positive, 1.0e8, 1.0e9, high])
    return tuple(float(value) for value in np.clip(raw, 0.0, high))


def _log_gradient(low_positive: float, high: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.geomspace(low_positive, high, 9))


def generate_fullrange_boundary18_samples(
    *,
    k_max: float = 1.0e11,
    random_count: int = 20,
    seed: int = 20260508,
) -> tuple[Boundary18Sample, ...]:
    """Return deterministic 18-variable samples spanning zero to ``k_max``."""

    if k_max <= 0.0:
        raise ValueError("k_max must be positive")
    if random_count < 0:
        raise ValueError("random_count must be non-negative")

    low_positive = 1.0e6
    uniform_values = (0.0, 1.0e6, 1.0e7, 1.0e8, 3.1622776601683795e8, 1.0e9, 1.0e10, k_max)
    samples: list[Boundary18Sample] = []
    for value in uniform_values:
        clipped = min(max(value, 0.0), k_max)
        samples.append(
            Boundary18Sample(
                f"uniform_{_safe_label(clipped)}",
                tuple([clipped] * 18),
                "Uniform released-rotation stiffness over the full design interval.",
            )
        )

    orientation_pairs = (
        (k_max, 0.0),
        (0.0, k_max),
        (1.0e9, 1.0e8),
        (1.0e8, 1.0e9),
        (1.0e10, 1.0e8),
        (1.0e8, 1.0e10),
        (1.0e10, 1.0e9),
        (1.0e9, 1.0e10),
    )
    for x_value, y_value in orientation_pairs:
        x = min(max(x_value, 0.0), k_max)
        y = min(max(y_value, 0.0), k_max)
        samples.append(
            Boundary18Sample(
                f"orient_x_{_safe_label(x)}_y_{_safe_label(y)}",
                tuple([x] * 9 + [y] * 9),
                "All x boundaries and all y boundaries use different stiffness levels.",
            )
        )

    center = _center_values(low_positive, k_max)
    edge = _edge_values(low_positive, k_max)
    gradient = _log_gradient(low_positive, k_max)
    reverse_gradient = tuple(reversed(gradient))
    sparse_center = tuple([0.0, 0.0, 0.0, 1.0e9, k_max, 1.0e9, 0.0, 0.0, 0.0])
    sparse_edges = tuple([k_max, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_max])
    structured = (
        ("center_zero_to_high", center + center, "Center boundaries are stiff; outer internal boundaries may remain hinged."),
        ("edge_zero_to_high", edge + edge, "Outer internal boundaries are stiff; center boundary may remain hinged."),
        ("x_gradient_y_low", gradient + tuple([1.0e8] * 9), "x boundaries follow a log gradient while y boundaries remain low."),
        ("x_low_y_gradient", tuple([1.0e8] * 9) + gradient, "y boundaries follow a log gradient while x boundaries remain low."),
        ("opposed_log_gradients", gradient + reverse_gradient, "x and y boundaries follow opposite log gradients."),
        ("sparse_center_high", sparse_center + sparse_center, "Only the middle region is assigned very high stiffness."),
        ("sparse_edges_high", sparse_edges + sparse_edges, "Only the outer internal boundaries are assigned very high stiffness."),
    )
    for name, values, description in structured:
        samples.append(Boundary18Sample(name, values, description))

    rng = np.random.default_rng(seed)
    for index in range(random_count):
        mask_zero = rng.random(18) < 0.20
        positive = 10.0 ** rng.uniform(np.log10(low_positive), np.log10(k_max), size=18)
        values = np.where(mask_zero, 0.0, positive)
        samples.append(
            Boundary18Sample(
                f"random_zero_log_{index + 1:02d}",
                tuple(float(value) for value in values),
                "Random sample with explicit zero values and log-uniform positive stiffness values.",
            )
        )

    return tuple(samples)


def _component_indices(labels: tuple[str, ...], names: tuple[str, ...]) -> tuple[int, ...]:
    wanted = set(names)
    return tuple(index for index, label in enumerate(labels) if label in wanted)


def _update_metric(
    metrics: dict[str, Any],
    value: float,
    key: str,
    cid_key: str,
    cid: str,
) -> None:
    if value > float(metrics[key]):
        metrics[key] = float(value)
        metrics[cid_key] = cid


def summarize_connector_relative_motion(case, response: np.ndarray, omega: float, *, cid_prefix: str) -> dict[str, Any]:
    """Return max relative-motion envelopes from recovered connector ``delta_hat``."""

    connectors = build_case_hinge_pair_connectors(case, cid_prefix=cid_prefix)
    recovered = recover_connector_response(
        np.asarray(response).reshape(-1),
        omega=float(omega),
        connectors=connectors,
    )
    metrics: dict[str, Any] = {
        "max_relative_uz_envelope": 0.0,
        "max_relative_uz_cid": "",
        "max_relative_translation_envelope": 0.0,
        "max_relative_translation_cid": "",
        "max_relative_rotation_envelope": 0.0,
        "max_relative_rotation_cid": "",
        "max_released_relative_rotation_envelope": 0.0,
        "max_released_relative_rotation_cid": "",
    }
    for connector in connectors:
        item = recovered[connector.cid]
        delta_hat = np.asarray(item["delta_hat"]).reshape(-1)
        labels = tuple(connector.labels)
        uz_indices = _component_indices(labels, ("uz",))
        translation_indices = _component_indices(labels, ("ux", "uy", "uz"))
        rotation_indices = _component_indices(labels, ("rx", "ry", "rz"))
        released_indices = tuple(connector.meta.get("released_retained_indices", ()))

        relative_uz, _ = harmonic_vector_norm_envelope(delta_hat[list(uz_indices)])
        relative_translation, _ = harmonic_vector_norm_envelope(delta_hat[list(translation_indices)])
        relative_rotation, _ = harmonic_vector_norm_envelope(delta_hat[list(rotation_indices)])
        released_rotation, _ = harmonic_vector_norm_envelope(delta_hat[list(released_indices)])

        _update_metric(metrics, relative_uz, "max_relative_uz_envelope", "max_relative_uz_cid", connector.cid)
        _update_metric(
            metrics,
            relative_translation,
            "max_relative_translation_envelope",
            "max_relative_translation_cid",
            connector.cid,
        )
        _update_metric(
            metrics,
            relative_rotation,
            "max_relative_rotation_envelope",
            "max_relative_rotation_cid",
            connector.cid,
        )
        _update_metric(
            metrics,
            released_rotation,
            "max_released_relative_rotation_envelope",
            "max_released_relative_rotation_cid",
            connector.cid,
        )
    return metrics


def _plot_tradeoff(rows: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    mean_heave = np.array([float(row["mean_heave"]) for row in rows])
    bending = np.array([float(row["max_connector_bending_envelope"]) for row in rows])
    rotation = np.array([float(row["max_released_relative_rotation_envelope"]) for row in rows])
    pareto = np.array([row["is_pareto"] in (True, "True", "true", "1") for row in rows])

    output_root.mkdir(parents=True, exist_ok=True)
    path_rotation = output_root / "boundary18_fullrange_mean_heave_rotation_bending.png"
    path_bending = output_root / "boundary18_fullrange_mean_heave_bending_rotation.png"

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    scatter = ax.scatter(mean_heave, rotation, c=bending, s=64, cmap="viridis", alpha=0.82)
    ax.scatter(
        mean_heave[pareto],
        rotation[pareto],
        facecolors="none",
        edgecolors="#d9480f",
        s=128,
        linewidths=1.5,
        label="3-objective Pareto",
    )
    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max released relative rotation (rad)")
    ax.set_title("boundary18 full-range DOE: motion vs relative rotation")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("max connector bending envelope")
    fig.savefig(path_rotation, dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    scatter = ax.scatter(mean_heave, bending, c=rotation, s=64, cmap="magma_r", alpha=0.82)
    ax.scatter(
        mean_heave[pareto],
        bending[pareto],
        facecolors="none",
        edgecolors="#1864ab",
        s=128,
        linewidths=1.5,
        label="3-objective Pareto",
    )
    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max connector bending envelope")
    ax.set_title("boundary18 full-range DOE: motion vs connector bending")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("max released relative rotation (rad)")
    fig.savefig(path_bending, dpi=240)
    plt.close(fig)
    return path_rotation, path_bending


def _select_representatives(rows: list[dict[str, Any]]) -> RepresentativeSet:
    by_label = {str(row["design_label"]): row for row in rows}
    pareto_rows = [row for row in rows if row["is_pareto"] in (True, "True", "true", "1")]
    candidates = pareto_rows if pareto_rows else rows

    selected: list[str] = []
    paper_anchor_labels = (
        "uniform_0",
        "uniform_1p00e07",
        "uniform_1p00e08",
        "uniform_3p16e08",
        "uniform_1p00e09",
        "orient_x_1p00e09_y_1p00e08",
        "orient_x_1p00e10_y_1p00e09",
        "orient_x_1p00e11_y_0",
        f"uniform_{_safe_label(1.0e11)}",
    )
    for label in paper_anchor_labels:
        if label in by_label:
            selected.append(label)

    for key in (
        "mean_heave",
        "max_released_relative_rotation_envelope",
        "max_connector_bending_envelope",
    ):
        selected.append(str(min(rows, key=lambda row: float(row[key]))["design_label"]))

    metrics = ("mean_heave", "max_released_relative_rotation_envelope", "max_connector_bending_envelope")
    values = np.array([[float(row[metric]) for metric in metrics] for row in candidates], dtype=float)
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    scale = np.where(upper > lower, upper - lower, 1.0)
    distance = np.linalg.norm((values - lower) / scale, axis=1)
    selected.append(str(candidates[int(np.argmin(distance))]["design_label"]))

    compact = []
    for label in selected:
        if label not in compact:
            compact.append(label)
    return RepresentativeSet(labels=tuple(compact))


def _plot_representative_stiffness(rows: list[dict[str, Any]], representatives: RepresentativeSet, output_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by_label = {str(row["design_label"]): row for row in rows}
    labels = [label for label in representatives.labels if label in by_label]
    matrix = []
    yticklabels = []
    for label in labels:
        row = by_label[label]
        values = [float(row[f"k_{group}"]) for group in BOUNDARY18_GROUP_NAMES]
        matrix.append(np.log10(np.asarray(values, dtype=float) + 1.0))
        yticklabels.append(label)
    data = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, max(3.4, 0.58 * len(labels) + 1.6)), constrained_layout=True)
    image = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0.0, vmax=np.log10(1.0e11 + 1.0))
    ax.set_xticks(np.arange(18))
    ax.set_xticklabels(
        [f"x{i}" for i in range(1, 10)] + [f"y{i}" for i in range(1, 10)],
        rotation=0,
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("boundary stiffness variable")
    ax.set_title("representative boundary18 stiffness distributions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(k + 1)")

    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "boundary18_fullrange_representative_stiffness_profiles.png"
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)
    return figure_path


def run_doe(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    response_root = output_root / "responses"
    response_root.mkdir(parents=True, exist_ok=True)

    samples = generate_fullrange_boundary18_samples(
        k_max=args.k_max,
        random_count=args.random_count,
        seed=args.seed,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    base_case = build_complex_hinge_10x10_case(
        args.data_root,
        k_hinge=args.coupling_stiffness,
    )
    groups = build_hinge_design_groups(base_case, "continuous_boundary")
    group_names = [group.name for group in groups]

    design_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[{sample_index}/{len(samples)}] solving {sample.name}", flush=True)
        design = BoundaryStiffnessDesign(
            values=sample.value_by_group(),
            grouping="continuous_boundary",
            parameter="released_dof_stiffness",
            coupling_stiffness=args.coupling_stiffness,
            label=sample.name,
            meta={"sample_description": sample.description},
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
                "scenario_label": "boundary18_fullrange_single_frequency",
            },
            heave_grid=solved.heave_grid_merged,
            cid_prefix=sample.name,
        )
        relative_metrics = summarize_connector_relative_motion(
            case,
            solved.response,
            solved.omega,
            cid_prefix=f"{sample.name}_delta",
        )

        summary_row = evaluation.summary_row()
        summary_row.update(relative_metrics)
        summary_row["sample_index"] = sample_index
        summary_row["solve_elapsed_s"] = elapsed
        summary_row["response_path"] = str(response_path)
        summary_row["heave_grid_path"] = str(heave_grid_path)
        summary_rows.append(summary_row)

    objectives = (
        MetricObjective("mean_heave", "mean_heave", minimize=True),
        MetricObjective("released_rotation", "max_released_relative_rotation_envelope", minimize=True),
        MetricObjective("connector_bending", "max_connector_bending_envelope", minimize=True),
    )
    pareto_rows = mark_pareto_rows(summary_rows, objectives)

    representatives = _select_representatives(pareto_rows)
    representative_rows = [
        row
        for row in pareto_rows
        if str(row["design_label"]) in set(representatives.labels)
    ]

    design_values_path = output_root / "boundary18_fullrange_design_values.csv"
    summary_path = output_root / "boundary18_fullrange_summary.csv"
    pareto_path = output_root / "boundary18_fullrange_pareto_summary.csv"
    representative_path = output_root / "boundary18_fullrange_representative_designs.csv"
    _write_csv(design_values_path, design_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(pareto_path, pareto_rows)
    _write_csv(representative_path, representative_rows)
    rotation_figure, bending_figure = _plot_tradeoff(pareto_rows, output_root)
    stiffness_figure = _plot_representative_stiffness(pareto_rows, representatives, output_root)

    return {
        "sample_count": len(samples),
        "pareto_count": sum(1 for row in pareto_rows if row["is_pareto"]),
        "summary_path": summary_path,
        "pareto_path": pareto_path,
        "representative_path": representative_path,
        "design_values_path": design_values_path,
        "rotation_figure": rotation_figure,
        "bending_figure": bending_figure,
        "stiffness_figure": stiffness_figure,
        "representatives": ",".join(representatives.labels),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-range 18-boundary stiffness DOE at one frequency.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", default="/Users/yongkang/data/DM-FEM2D")
    parser.add_argument("--k-max", type=float, default=1.0e11)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--coupling-stiffness", type=float, default=1.0e10)
    parser.add_argument("--frequency-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_doe(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
