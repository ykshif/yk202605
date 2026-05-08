"""Build PDF figures for the paper result-and-discussion section."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "paper_results_discussion"
NECESSITY_ROOT = REPO_ROOT / "results" / "notebook_hinge_10x10_20260501"
PITCH_SWEEP_ROOT = REPO_ROOT / "results" / "complex_hinge_10x10_pitch_stiffness_sweep"
BOUNDARY18_ROOT = REPO_ROOT / "results" / "boundary18_fullrange_single_frequency"
FOCUSED18_ROOT = REPO_ROOT / "results" / "boundary18_focused_refinement"
LOW_STIFFNESS_ROOT = REPO_ROOT / "results" / "low_stiffness_sensitivity"
MODULE_COUNT_ROOT = REPO_ROOT / "results" / "module_count_comparison"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


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


def _copy_figure_pair(source_pdf: Path, destination_pdf: Path) -> Path:
    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_pdf, destination_pdf)
    source_png = source_pdf.with_suffix(".png")
    if source_png.exists():
        shutil.copyfile(source_png, destination_pdf.with_suffix(".png"))
    return destination_pdf


def _label_from_k(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(round(np.log10(value)))
    if np.isclose(value, 10.0 ** exponent):
        return f"$10^{{{exponent}}}$"
    return f"{value:.2g}"


def _is_true(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _display_design_label(label: str) -> str:
    replacements = {
        "uniform_0": "hinged\n$k=0$",
        "uniform_1p00e07": "uniform\n$10^7$",
        "uniform_1p00e08": "uniform\n$10^8$",
        "uniform_3p16e08": "uniform\n$3.16\\times10^8$",
        "uniform_1p00e09": "uniform\n$10^9$",
        "uniform_1p00e10": "uniform\n$10^{10}$",
        "uniform_1p00e11": "uniform\n$10^{11}$",
        "orient_x_1p00e09_y_1p00e08": "$x:10^9$\n$y:10^8$",
        "orient_x_1p00e10_y_1p00e09": "$x:10^{10}$\n$y:10^9$",
        "orient_x_1p00e11_y_0": "$x:10^{11}$\n$y:0$",
        "x_low_y_gradient": "$x:10^8$\n$y$ gradient",
    }
    return replacements.get(label, label.replace("_", "\n"))


def _row_by_label(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["design_label"]): row for row in rows}


def build_necessity_figures(output_root: Path) -> list[Path]:
    plt = _ensure_matplotlib()
    figure_root = output_root / "figures" / "section_3_1"
    rows = _read_csv(NECESSITY_ROOT / "optimization_necessity_uniform_scheme_summary.csv")
    rows = sorted(rows, key=lambda row: _float(row, "pitch_stiffness"))
    labels = [_label_from_k(_float(row, "pitch_stiffness")) for row in rows]
    mean_heave = np.array([_float(row, "mean_heave") for row in rows])
    bending = np.array([_float(row, "max_connector_bending_envelope") for row in rows])
    rotation = np.array([_float(row, "max_released_relative_rotation_envelope") for row in rows])
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    axes[0].bar(labels, mean_heave, color="#2b8a3e")
    axes[0].set_ylabel("mean heave amplitude (m)")
    axes[1].bar(labels, bending, color="#e67700")
    axes[1].set_ylabel("max connector bending envelope")
    axes[2].bar(labels, rotation, color="#5f3dc4")
    axes[2].set_ylabel("max released relative rotation (rad)")
    for ax in axes:
        ax.set_xlabel("uniform released stiffness")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    fig.suptitle("Uniform connection schemes: necessity of stiffness optimization")
    paths.append(_save(fig, figure_root / "fig3_1_uniform_scheme_metrics.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    scatter = ax.scatter(mean_heave, rotation, c=bending, s=92, cmap="viridis")
    ax.plot(mean_heave, rotation, color="#495057", linewidth=1.0, alpha=0.55)
    for label, x, y in zip(labels, mean_heave, rotation):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 5))
    ax.set_xlabel("mean heave amplitude (m)")
    ax.set_ylabel("max released relative rotation (rad)")
    ax.set_title("Motion-relative-rotation tradeoff")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("max connector bending envelope")
    paths.append(_save(fig, figure_root / "fig3_1_heave_rotation_tradeoff.pdf"))
    plt.close(fig)

    heatmap_items = []
    for row in rows:
        k = _float(row, "pitch_stiffness")
        if k == 0:
            label = "k=0"
            filename = "heave_grid_merged_k_pitch0.npy"
        else:
            label = f"k={_label_from_k(k)}"
            filename = f"heave_grid_merged_k_pitch{int(k):.0e}".replace("+", "")
            filename = filename.replace("e0", "e").replace("e+0", "e") + ".npy"
            # Existing cache files use labels such as k_pitch1e8.
            filename = {
                1.0e8: "heave_grid_merged_k_pitch1e8.npy",
                1.0e9: "heave_grid_merged_k_pitch1e9.npy",
                1.0e10: "heave_grid_merged_k_pitch1e10.npy",
            }.get(k, filename)
        path = PITCH_SWEEP_ROOT / filename
        if path.exists():
            heatmap_items.append((label, np.load(path)))
    if heatmap_items:
        vmin = min(float(grid.min()) for _, grid in heatmap_items)
        vmax = max(float(grid.max()) for _, grid in heatmap_items)
        fig, axes = plt.subplots(1, len(heatmap_items), figsize=(4.1 * len(heatmap_items), 3.8), constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for ax, (label, grid) in zip(axes, heatmap_items):
            image = ax.imshow(grid, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax, extent=[0, 300, 0, 300])
            ax.set_title(label)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.88)
            cbar.set_label("heave amplitude (m)")
        fig.suptitle("Heave fields for uniform connection schemes")
        paths.append(_save(fig, figure_root / "fig3_1_uniform_scheme_heave_fields.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for label, grid in heatmap_items:
            x = np.linspace(0.0, 300.0, grid.shape[1])
            ax.plot(x, grid[grid.shape[0] // 2, :], linewidth=1.6, label=label)
        ax.set_xlabel("x at platform centerline (m)")
        ax.set_ylabel("heave amplitude (m)")
        ax.set_title("Centerline heave comparison")
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
        ax.legend(frameon=False)
        paths.append(_save(fig, figure_root / "fig3_1_uniform_scheme_centerline.pdf"))
        plt.close(fig)
    return paths


def build_boundary18_figures(output_root: Path) -> list[Path]:
    plt = _ensure_matplotlib()
    figure_root = output_root / "figures" / "section_3_2"
    rows = _read_csv(BOUNDARY18_ROOT / "boundary18_fullrange_pareto_summary.csv")
    rep_rows = _read_csv(BOUNDARY18_ROOT / "boundary18_fullrange_representative_designs.csv")
    paths: list[Path] = []

    def draw_grid_parameterization(ax, *, segment_level: bool) -> None:
        n = 10
        for index in range(n + 1):
            ax.plot([0, n], [index, index], color="#dee2e6", linewidth=0.75, zorder=0)
            ax.plot([index, index], [0, n], color="#dee2e6", linewidth=0.75, zorder=0)
        if segment_level:
            cmap_x = plt.get_cmap("Blues")
            cmap_y = plt.get_cmap("Oranges")
            for boundary in range(1, n):
                for segment in range(n):
                    ax.plot(
                        [boundary, boundary],
                        [segment + 0.06, segment + 0.94],
                        color=cmap_x(0.35 + 0.55 * segment / (n - 1)),
                        linewidth=2.4,
                    )
            for boundary in range(1, n):
                for segment in range(n):
                    ax.plot(
                        [segment + 0.06, segment + 0.94],
                        [boundary, boundary],
                        color=cmap_y(0.35 + 0.55 * segment / (n - 1)),
                        linewidth=2.4,
                    )
            ax.set_title("180 segment-level variables")
            ax.text(
                0.1,
                -0.9,
                "each 30 m hinge segment has its own stiffness\n90 x-segments + 90 y-segments",
                fontsize=9,
                va="top",
            )
        else:
            for boundary in range(1, n):
                ax.plot([boundary, boundary], [0, n], color="#1971c2", linewidth=3.0)
                ax.text(boundary, n + 0.35, f"$x_{boundary}$", ha="center", va="bottom", fontsize=8)
            for boundary in range(1, n):
                ax.plot([0, n], [boundary, boundary], color="#e67700", linewidth=3.0)
                ax.text(-0.35, boundary, f"$y_{boundary}$", ha="right", va="center", fontsize=8)
            ax.set_title("18 continuous-boundary variables")
            ax.text(
                0.1,
                -0.9,
                "one variable controls one full internal boundary\n18 variables = 9 vertical + 9 horizontal",
                fontsize=9,
                va="top",
            )
        ax.set_xlim(-0.75, n + 0.75)
        ax.set_ylim(-1.4, n + 0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    draw_grid_parameterization(axes[0], segment_level=False)
    draw_grid_parameterization(axes[1], segment_level=True)
    fig.suptitle("10x10 hinge-stiffness parameterization")
    paths.append(_save(fig, figure_root / "fig3_2_parameterization_18_vs_180.pdf"))
    plt.close(fig)

    focused_main_figures = (
        (
            LOW_STIFFNESS_ROOT / "figures" / "low_stiffness_uniform_metrics.pdf",
            figure_root / "fig3_2_low_stiffness_uniform_metrics.pdf",
        ),
        (
            FOCUSED18_ROOT / "figures" / "focused18_pareto_projection.pdf",
            figure_root / "fig3_2_focused18_pareto_projection.pdf",
        ),
        (
            FOCUSED18_ROOT / "figures" / "focused18_nonuniform_gain_vs_uniform.pdf",
            figure_root / "fig3_2_focused18_nonuniform_gain_vs_uniform.pdf",
        ),
        (
            FOCUSED18_ROOT / "figures" / "focused18_representative_stiffness_profiles.pdf",
            figure_root / "fig3_2_focused18_representative_stiffness_profiles.pdf",
        ),
    )
    if all(source.exists() for source, _destination in focused_main_figures):
        for source, destination in focused_main_figures:
            paths.append(_copy_figure_pair(source, destination))
        return paths

    mean_heave = np.array([_float(row, "mean_heave") for row in rows])
    bending = np.array([_float(row, "max_connector_bending_envelope") for row in rows])
    rotation = np.array([_float(row, "max_released_relative_rotation_envelope") for row in rows])
    pareto = np.array([_is_true(row["is_pareto"]) for row in rows])
    labels_all = np.array([str(row["design_label"]) for row in rows])
    uniform = np.array([label.startswith("uniform") for label in labels_all])
    by_label = _row_by_label(rows)
    key_labels = [
        "uniform_0",
        "uniform_1p00e07",
        "uniform_1p00e08",
        "uniform_1p00e09",
        "orient_x_1p00e09_y_1p00e08",
        "uniform_1p00e11",
    ]
    key_rows = [by_label[label] for label in key_labels if label in by_label]
    key_table_path = output_root / "section_3_2_key_design_metrics.csv"
    key_table_path.parent.mkdir(parents=True, exist_ok=True)
    with key_table_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "design_label",
            "design_dimension",
            "mean_heave",
            "max_released_relative_rotation_envelope",
            "max_connector_bending_envelope",
            "max_connector_shear_envelope",
            "is_pareto",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(key_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    uniform_order = np.argsort([_float(row, "boundary_stiffness_mean") for row in rows])
    uniform_order = [index for index in uniform_order if uniform[index]]

    uniform_pareto = pareto & uniform
    nonuniform_pareto = pareto & ~uniform
    axes[0].scatter(mean_heave, rotation, color="#adb5bd", s=36, alpha=0.38, label="DOE samples")
    axes[0].scatter(
        mean_heave[uniform_pareto],
        rotation[uniform_pareto],
        color="#f08c00",
        s=74,
        edgecolors="#212529",
        linewidths=0.45,
        label="uniform Pareto",
    )
    axes[0].scatter(
        mean_heave[nonuniform_pareto],
        rotation[nonuniform_pareto],
        color="#1c7ed6",
        marker="D",
        s=58,
        edgecolors="#212529",
        linewidths=0.45,
        label="non-uniform Pareto",
    )
    axes[0].plot(mean_heave[uniform_order], rotation[uniform_order], color="#495057", linewidth=1.3, label="uniform path")
    axes[0].set_xlabel("mean heave amplitude (m)")
    axes[0].set_ylabel("max released relative rotation (rad)")
    axes[0].set_title("Motion vs relative rotation")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    rotation_offsets = {
        "uniform_0": (10, 6),
        "uniform_1p00e07": (-58, -10),
        "uniform_1p00e08": (-36, 12),
        "uniform_1p00e09": (8, -18),
        "orient_x_1p00e09_y_1p00e08": (8, 6),
        "uniform_1p00e11": (8, 8),
    }
    for label in key_labels:
        if label not in by_label:
            continue
        row = by_label[label]
        axes[0].annotate(
            _display_design_label(label),
            (_float(row, "mean_heave"), _float(row, "max_released_relative_rotation_envelope")),
            textcoords="offset points",
            xytext=rotation_offsets.get(label, (6, 4)),
            fontsize=7.5,
        )

    axes[1].scatter(mean_heave, bending / 1.0e6, color="#adb5bd", s=36, alpha=0.45, label="DOE samples")
    axes[1].scatter(
        mean_heave[uniform_pareto],
        bending[uniform_pareto] / 1.0e6,
        color="#f08c00",
        s=74,
        edgecolors="#212529",
        linewidths=0.45,
        label="uniform Pareto",
    )
    axes[1].scatter(
        mean_heave[nonuniform_pareto],
        bending[nonuniform_pareto] / 1.0e6,
        color="#1c7ed6",
        marker="D",
        s=58,
        edgecolors="#212529",
        linewidths=0.45,
        label="non-uniform Pareto",
    )
    axes[1].plot(mean_heave[uniform_order], bending[uniform_order] / 1.0e6, color="#495057", linewidth=1.3)
    axes[1].set_xlabel("mean heave amplitude (m)")
    axes[1].set_ylabel("max connector bending envelope ($\\times 10^6$)")
    axes[1].set_title("Motion vs connector bending")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.7)
    bending_offsets = {
        "uniform_0": (8, 0),
        "uniform_1p00e07": (8, -16),
        "uniform_1p00e08": (8, 5),
        "uniform_1p00e09": (8, 0),
        "orient_x_1p00e09_y_1p00e08": (8, -18),
        "uniform_1p00e11": (8, 4),
    }
    for label in key_labels:
        if label not in by_label:
            continue
        row = by_label[label]
        axes[1].annotate(
            _display_design_label(label),
            (_float(row, "mean_heave"), _float(row, "max_connector_bending_envelope") / 1.0e6),
            textcoords="offset points",
            xytext=bending_offsets.get(label, (6, 4)),
            fontsize=7.5,
        )
    fig.suptitle("18-variable DOE and Pareto screening with baseline paths")
    paths.append(_save(fig, figure_root / "fig3_2_pareto_projection_with_baselines.pdf"))
    plt.close(fig)

    if key_rows:
        metric_specs = [
            ("mean_heave", "mean heave (m)", "#2f9e44", 1.0),
            ("max_released_relative_rotation_envelope", "max released relative rotation (rad)", "#5f3dc4", 1.0),
            ("max_connector_bending_envelope", "max connector bending envelope ($\\times10^6$)", "#e67700", 1.0e-6),
        ]
        x = np.arange(len(key_rows))
        tick_labels = [_display_design_label(str(row["design_label"])) for row in key_rows]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
        for ax, (metric, ylabel, color, scale) in zip(axes, metric_specs):
            values = np.array([_float(row, metric) * scale for row in key_rows])
            ax.bar(x, values, color=color, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
            best_index = int(np.argmin(values))
            ax.scatter([best_index], [values[best_index]], s=72, facecolors="none", edgecolors="#212529", linewidths=1.2)
        fig.suptitle("Key 10x10 designs: what the 18-variable study actually changes")
        paths.append(_save(fig, figure_root / "fig3_2_key_design_objective_comparison.pdf"))
        plt.close(fig)

    group_names = [f"x_boundary_{index:02d}" for index in range(1, 10)] + [
        f"y_boundary_{index:02d}" for index in range(1, 10)
    ]
    matrix = []
    labels = []
    for row in rep_rows:
        labels.append(_display_design_label(str(row["design_label"])))
        values = np.array([_float(row, f"k_{name}") for name in group_names])
        matrix.append(np.log10(values + 1.0))
    fig, ax = plt.subplots(figsize=(10.6, max(4.2, 0.48 * len(labels) + 1.8)), constrained_layout=True)
    image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="cividis", vmin=0.0, vmax=np.log10(1.0e11 + 1.0))
    ax.set_xticks(np.arange(18))
    ax.set_xticklabels([f"x{i}" for i in range(1, 10)] + [f"y{i}" for i in range(1, 10)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("boundary stiffness variable")
    ax.set_title("Representative 18-variable stiffness distributions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(k + 1)")
    paths.append(_save(fig, figure_root / "fig3_2_representative_stiffness_profiles.pdf"))
    plt.close(fig)

    heave_labels = [
        "uniform_0",
        "uniform_1p00e07",
        "orient_x_1p00e09_y_1p00e08",
        "uniform_1p00e09",
    ]
    heatmaps = []
    for label in heave_labels:
        path = BOUNDARY18_ROOT / "responses" / f"heave_grid_{label}.npy"
        if path.exists():
            heatmaps.append((_display_design_label(label), np.load(path)))
    if heatmaps:
        vmin = min(float(grid.min()) for _, grid in heatmaps)
        vmax = max(float(grid.max()) for _, grid in heatmaps)
        fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.1 * len(heatmaps), 3.8), constrained_layout=True)
        axes = np.atleast_1d(axes)
        image = None
        for ax, (label, grid) in zip(axes, heatmaps):
            image = ax.imshow(grid, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax, extent=[0, 300, 0, 300])
            ax.set_title(label)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.88)
            cbar.set_label("heave amplitude (m)")
        fig.suptitle("Representative 18-variable heave fields")
        paths.append(_save(fig, figure_root / "fig3_2_representative_heave_fields.pdf"))
        plt.close(fig)
    return paths


def build_module_count_figures(output_root: Path) -> list[Path]:
    plt = _ensure_matplotlib()
    figure_root = output_root / "figures" / "section_3_3"
    paths: list[Path] = []
    inventory_path = MODULE_COUNT_ROOT / "module_count_design_space_and_inputs.csv"
    if not inventory_path.exists():
        return paths
    rows = _read_csv(inventory_path)
    first_stiffness = rows[0]["released_dof_stiffness"] if rows else "0.0"
    base_rows = [row for row in rows if row["released_dof_stiffness"] == first_stiffness]
    n = np.array([int(row["modules_per_side"]) for row in base_rows])
    module_count = np.array([int(row["module_count"]) for row in base_rows])
    hinge_lines = np.array([int(row["hinge_line_count"]) for row in base_rows])
    connector_pairs = np.array([int(row["connector_pair_count"]) for row in base_rows])
    boundary_dim = np.array([int(row["continuous_boundary_dimension"]) for row in base_rows])
    line_dim = np.array([int(row["segment_line_dimension"]) for row in base_rows])
    structural_nodes = np.array([int(row.get("structural_node_count", 0)) for row in base_rows])
    retained_hydro_dofs = np.array(
        [int(row.get("retained_hydrodynamic_dof_count", int(row["module_count"]) * 5)) for row in base_rows]
    )
    bem_problem_count = np.array(
        [
            int(row.get("single_frequency_bem_problem_count", int(row["module_count"]) * 6 + 1))
            for row in base_rows
        ]
    )
    module_mesh_side = np.array(
        [int(row.get("nodes_per_module_side", 7)) for row in base_rows]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(n, module_count, marker="o", label="modules")
    ax.plot(n, hinge_lines, marker="s", label="hinge lines")
    ax.plot(n, connector_pairs, marker="^", label="connector pairs")
    ax.plot(n, line_dim, marker="v", label="line variables")
    ax.plot(n, boundary_dim, marker="D", label="boundary variables")
    ax.set_xlabel("modules per side")
    ax.set_ylabel("count")
    ax.set_title("Module discretization changes design-space size")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    paths.append(_save(fig, figure_root / "fig3_3_module_count_design_space.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    axes[0].bar([f"{value}x{value}" for value in n], structural_nodes, color="#0b7285")
    axes[0].set_ylabel("structural nodes")
    axes[0].set_title("Condensed structural model size")
    axes[0].grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    for index, value in enumerate(module_mesh_side):
        axes[0].text(index, structural_nodes[index], f"{value}x{value}/module", ha="center", va="bottom", fontsize=8)
    axes[1].bar([f"{value}x{value}" for value in n], connector_pairs, color="#1971c2")
    axes[1].set_ylabel("connector node pairs")
    axes[1].set_title("Recovered connector-force scale")
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    paths.append(_save(fig, figure_root / "fig3_3_module_count_structural_and_connectors.pdf"))
    plt.close(fig)

    availability = np.array(
        [
            [
                str(row["mass_matrix_exists"]).lower() == "true",
                str(row["stiffness_matrix_exists"]).lower() == "true",
                str(row["hydrodynamic_exists"]).lower() == "true",
            ]
            for row in base_rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    image = ax.imshow(availability, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["mass", "stiffness", "hydro"])
    ax.set_yticks(np.arange(len(base_rows)))
    ax.set_yticklabels([f"{row['modules_per_side']}x{row['modules_per_side']}" for row in base_rows])
    ax.set_title("Available inputs for module-count response comparison")
    for i in range(availability.shape[0]):
        for j in range(availability.shape[1]):
            ax.text(j, i, "yes" if availability[i, j] else "missing", ha="center", va="center", fontsize=9)
    paths.append(_save(fig, figure_root / "fig3_3_module_count_input_availability.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    x_labels = [f"{value}x{value}" for value in n]
    ax.bar(x_labels, bem_problem_count, color="#0b7285", label="BEM problems")
    ax.plot(x_labels, retained_hydro_dofs, color="#c92a2a", marker="o", label="retained hydro DOFs")
    ax.set_ylabel("count")
    ax.set_title("Hydrodynamic generation workload")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    ax.legend(frameon=False)
    paths.append(_save(fig, figure_root / "fig3_3_module_count_hydro_workload.pdf"))
    plt.close(fig)

    response_path = MODULE_COUNT_ROOT / "module_count_response_summary.csv"
    if response_path.exists():
        response_rows = _read_csv(response_path)
        solved_modules = sorted({int(row["modules_per_side"]) for row in response_rows})
        stiffness_values = sorted({float(row["released_dof_stiffness"]) for row in response_rows})
        colors = {0: "#2f9e44", 5: "#0b7285", 10: "#1971c2", 15: "#e67700", 20: "#862e9c"}

        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)
        metric_specs = [
            ("mean_heave", "mean heave amplitude (m)"),
            ("max_connector_bending_envelope", "max connector bending envelope"),
            ("max_released_moment_envelope", "released moment envelope"),
        ]
        for ax, (metric, ylabel) in zip(axes, metric_specs):
            for module in solved_modules:
                rows_for_module = [
                    row
                    for row in response_rows
                    if int(row["modules_per_side"]) == module
                ]
                rows_for_module = sorted(rows_for_module, key=lambda row: float(row["released_dof_stiffness"]))
                ax.plot(
                    [_float(row, "released_dof_stiffness") for row in rows_for_module],
                    [_float(row, metric) for row in rows_for_module],
                    marker="o",
                    label=f"{module}x{module}",
                    color=colors.get(module, None),
                )
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlabel("released rotational stiffness")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#d9d9d9", linewidth=0.7)
        axes[0].legend(frameon=False)
        fig.suptitle("Solved response metrics for available module-count cases")
        paths.append(_save(fig, figure_root / "fig3_3_module_count_available_response_metrics.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        for module in solved_modules:
            rows_for_module = [
                row
                for row in response_rows
                if int(row["modules_per_side"]) == module
            ]
            rows_for_module = sorted(rows_for_module, key=lambda row: float(row["released_dof_stiffness"]))
            if not rows_for_module:
                continue
            reference = _float(rows_for_module[0], "mean_heave")
            ax.plot(
                [_float(row, "released_dof_stiffness") for row in rows_for_module],
                [
                    (_float(row, "mean_heave") / reference - 1.0) * 100.0
                    if reference
                    else 0.0
                    for row in rows_for_module
                ],
                marker="o",
                label=f"{module}x{module}",
                color=colors.get(module, None),
            )
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("released rotational stiffness")
        ax.set_ylabel("mean-heave change from k=0 (%)")
        ax.set_title("Stiffness sensitivity depends on module discretization")
        ax.grid(True, color="#d9d9d9", linewidth=0.7)
        ax.legend(frameon=False)
        paths.append(_save(fig, figure_root / "fig3_3_module_count_normalized_heave_sensitivity.pdf"))
        plt.close(fig)

        heatmap_items = []
        response_root = MODULE_COUNT_ROOT / "responses"
        for module in solved_modules:
            rows_for_module = [
                row
                for row in response_rows
                if int(row["modules_per_side"]) == module
            ]
            for target_k in (0.0, max(stiffness_values)):
                chosen = min(
                    rows_for_module,
                    key=lambda row: abs(float(row["released_dof_stiffness"]) - target_k),
                )
                path = Path(chosen.get("heave_grid_path", ""))
                if not path.exists():
                    path = response_root / f"heave_grid_{module}x{module}_k{chosen['stiffness_label']}.npy"
                if path.exists():
                    heatmap_items.append(
                        (
                            f"{module}x{module}, k={_label_from_k(float(chosen['released_dof_stiffness']))}",
                            np.load(path),
                        )
                    )
        if heatmap_items:
            vmin = min(float(grid.min()) for _, grid in heatmap_items)
            vmax = max(float(grid.max()) for _, grid in heatmap_items)
            columns = 3 if len(heatmap_items) >= 6 else min(3, len(heatmap_items))
            rows_count = int(np.ceil(len(heatmap_items) / columns))
            fig, axes = plt.subplots(
                rows_count,
                columns,
                figsize=(3.6 * columns, 3.4 * rows_count),
                constrained_layout=True,
            )
            axes = np.asarray(axes).reshape(-1)
            image = None
            for ax, (label, grid) in zip(axes, heatmap_items):
                image = ax.imshow(
                    grid,
                    origin="upper",
                    extent=[0, 300, 0, 300],
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(label)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
            for ax in axes[len(heatmap_items):]:
                ax.axis("off")
            if image is not None:
                cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.86)
                cbar.set_label("heave amplitude (m)")
            fig.suptitle("Representative heave fields across module discretizations")
            paths.append(_save(fig, figure_root / "fig3_3_module_count_representative_heave_fields.pdf"))
            plt.close(fig)
    return paths


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    paths = []
    paths.extend(build_necessity_figures(output_root))
    paths.extend(build_boundary18_figures(output_root))
    paths.extend(build_module_count_figures(output_root))
    manifest = {
        "figure_count": len(paths),
        "figures": [str(path) for path in paths],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "paper_results_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
