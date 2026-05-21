"""Build a frequency- and mode-adaptive control-point density indicator.

This is a diagnostic workflow for the SEREP-ridge non-uniform module study.
It explains where more control points are needed by combining:

1. target-frequency response curvature from the U30 reference response;
2. structural modal curvature weighted by frequency-dependent modal participation;
3. local error/improvement checks against U10 and target-best NU10 layouts.

The next optimization step can replace the U30 response curvature with a
low-cost pilot response. For this step, the U30 response is intentionally used
to validate and interpret the indicator.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_serep_nonuniform_design_study as base  # noqa: E402
from offshore_energy_sim.hydrodynamics import omega_values_from_wavelengths  # noqa: E402
from offshore_energy_sim.reduction.dofs import reduce_matrix_dofs  # noqa: E402
from offshore_energy_sim.reduction.modal import transform_mass_matrix  # noqa: E402
from offshore_energy_sim.response.retained_dofs import retained_node_dof_series  # noqa: E402
from offshore_energy_sim.structure.matrix_io import read_abaqus_matrix_dense  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "results" / "frequency_mode_adaptive_density"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "frequency_mode_adaptive_density_report.md"
MODAL_CACHE = OUTPUT_ROOT / "modal_cache_first80.npz"

SWEEP_ROOT = REPO_ROOT / "results" / "serep_ridge_nonuniform_wavelength_sweep"
RESPONSE_ROOT = SWEEP_ROOT / "responses"
BEST_BY_WAVELENGTH_CSV = SWEEP_ROOT / "tables" / "wavelength_sweep_best_by_wavelength.csv"
LAYOUT_SUMMARY_CSV = SWEEP_ROOT / "tables" / "wavelength_sweep_layout_summary.csv"

BODY_LENGTH_M = 300.0
BODY_WIDTH_M = 60.0
WATER_DEPTH_M = 58.5
G = 9.81
TARGET_WAVELENGTHS_M = (120, 180, 240, 300)
ALL_SWEEP_WAVELENGTHS_M = (60, 90, 120, 150, 180, 210, 240, 270, 300)
TOTAL_NODES = 793
FULL_DOFS_PER_NODE = 6
RETAINED_DOFS_PER_NODE = 5
REMOVED_FULL_DOFS_ZERO_BASED = (5,)
CENTERLINE_START_NODE = 367
CENTERLINE_STOP_NODE = 427
VERTICAL_DOF_ZERO_BASED = 2
DEFAULT_MODE_COUNT = 80
RESPONSE_WEIGHT = 0.5


def file_uri(path: Path) -> str:
    return "file:///" + path.resolve().as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= 1.0e-14:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def curvature_magnitude(values: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    first = np.gradient(values, x_m)
    second = np.gradient(first, x_m)
    return np.abs(second)


def response_path(layout_id: str, wavelength_m: int) -> Path:
    return RESPONSE_ROOT / layout_id / f"{layout_id}_wavelength_{wavelength_m}m_response.npy"


def load_response(layout_id: str, wavelength_m: int) -> np.ndarray:
    path = response_path(layout_id, wavelength_m)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def centerline_vertical_complex(response: np.ndarray) -> np.ndarray:
    return retained_node_dof_series(
        response,
        start_node_one_based=CENTERLINE_START_NODE,
        stop_node_one_based=CENTERLINE_STOP_NODE,
        retained_dofs_per_node=RETAINED_DOFS_PER_NODE,
        dof_index_zero_based=VERTICAL_DOF_ZERO_BASED,
        column=0,
    )


def centerline_x() -> np.ndarray:
    sample = centerline_vertical_complex(load_response("U30_reference", TARGET_WAVELENGTHS_M[0]))
    return np.linspace(0.0, BODY_LENGTH_M, sample.size)


def load_layout_lengths() -> dict[str, tuple[float, ...]]:
    layouts: dict[str, tuple[float, ...]] = {}
    for row in read_csv(LAYOUT_SUMMARY_CSV):
        layouts[row["layout_id"]] = tuple(float(value) for value in row["module_lengths_m"].split())
    return layouts


def target_best_layouts() -> dict[int, str]:
    best: dict[int, str] = {}
    for row in read_csv(BEST_BY_WAVELENGTH_CSV):
        wavelength_m = int(row["wavelength_m"])
        if wavelength_m in TARGET_WAVELENGTHS_M:
            best[wavelength_m] = row["best_layout_id"]
    return best


def module_boundaries(lengths: tuple[float, ...]) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(lengths, dtype=float))])


def module_average(x_m: np.ndarray, values: np.ndarray, boundaries_m: np.ndarray) -> list[float]:
    averages = []
    for start, end in zip(boundaries_m[:-1], boundaries_m[1:]):
        if np.isclose(end, BODY_LENGTH_M):
            mask = (x_m >= start) & (x_m <= end)
        else:
            mask = (x_m >= start) & (x_m < end)
        if np.any(mask):
            averages.append(float(np.mean(values[mask])))
        else:
            averages.append(float(np.interp(0.5 * (start + end), x_m, values)))
    return averages


def load_or_compute_modes(mode_count: int, *, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if MODAL_CACHE.exists() and not force:
        data = np.load(MODAL_CACHE)
        eigenvalues = data["eigenvalues"]
        modes = data["modes"]
        mass = data["mass"]
        if modes.shape[1] >= mode_count:
            return eigenvalues[:mode_count], modes[:, :mode_count], mass

    from scipy.linalg import eigh

    print("Reading structural matrices...")
    paths = base.structural_paths()
    mass_full = read_abaqus_matrix_dense(paths.mass, dofs_per_node=FULL_DOFS_PER_NODE)
    stiffness_full = read_abaqus_matrix_dense(paths.stiffness, dofs_per_node=FULL_DOFS_PER_NODE)
    mass = transform_mass_matrix(
        reduce_matrix_dofs(mass_full, TOTAL_NODES, REMOVED_FULL_DOFS_ZERO_BASED),
        beta=0.0,
    )
    stiffness = reduce_matrix_dofs(stiffness_full, TOTAL_NODES, REMOVED_FULL_DOFS_ZERO_BASED)
    del mass_full, stiffness_full

    print(f"Solving first {mode_count} structural modes...")
    eigenvalues, modes = eigh(stiffness, mass, subset_by_index=[0, mode_count - 1])
    MODAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODAL_CACHE, eigenvalues=eigenvalues, modes=modes, mass=mass)
    return eigenvalues, modes, mass


def modal_centerline_curvatures(modes: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    curves = []
    for mode_index in range(modes.shape[1]):
        vertical = centerline_vertical_complex(modes[:, [mode_index]]).real
        max_abs = float(np.max(np.abs(vertical)))
        if max_abs > 0.0:
            vertical = vertical / max_abs
        curves.append(normalize(curvature_magnitude(vertical, x_m)))
    return np.asarray(curves)


def modal_participation(
    response: np.ndarray,
    modes: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    vector = response[:, 0]
    coordinates = modes.conj().T @ (mass @ vector)
    weights = np.abs(coordinates) ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones(modes.shape[1]) / modes.shape[1]
    return weights / total


def density_components(
    *,
    wavelength_m: int,
    modes: np.ndarray,
    mass: np.ndarray,
    modal_curvatures: np.ndarray,
    x_m: np.ndarray,
) -> dict[str, np.ndarray]:
    response = load_response("U30_reference", wavelength_m)
    vertical = centerline_vertical_complex(response)
    response_curvature = normalize(curvature_magnitude(np.abs(vertical), x_m))
    participation = modal_participation(response, modes, mass)
    modal_curvature = normalize(participation @ modal_curvatures)
    density = normalize(RESPONSE_WEIGHT * response_curvature + (1.0 - RESPONSE_WEIGHT) * modal_curvature)
    return {
        "response_curvature": response_curvature,
        "modal_curvature": modal_curvature,
        "density": density,
        "participation": participation,
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1.0e-14 or np.std(b) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_density_tables(
    *,
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    mass: np.ndarray,
    modal_curvatures: np.ndarray,
    x_m: np.ndarray,
) -> dict[str, Path]:
    best_layouts = target_best_layouts()
    layouts = load_layout_lengths()
    omegas = omega_values_from_wavelengths(ALL_SWEEP_WAVELENGTHS_M, WATER_DEPTH_M, G)
    omega_by_wavelength = {wl: float(omega) for wl, omega in zip(ALL_SWEEP_WAVELENGTHS_M, omegas)}

    density_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    modal_rows: list[dict[str, object]] = []

    for wavelength_m in TARGET_WAVELENGTHS_M:
        components = density_components(
            wavelength_m=wavelength_m,
            modes=modes,
            mass=mass,
            modal_curvatures=modal_curvatures,
            x_m=x_m,
        )
        response_curvature = components["response_curvature"]
        modal_curvature = components["modal_curvature"]
        density = components["density"]
        participation = components["participation"]

        ref = centerline_vertical_complex(load_response("U30_reference", wavelength_m))
        uniform = centerline_vertical_complex(load_response("uniform_U10", wavelength_m))
        best_layout = best_layouts[wavelength_m]
        best = centerline_vertical_complex(load_response(best_layout, wavelength_m))
        uniform_error = np.abs(np.abs(uniform) - np.abs(ref))
        best_error = np.abs(np.abs(best) - np.abs(ref))
        local_improvement = uniform_error - best_error

        for index, x_value in enumerate(x_m):
            density_rows.append(
                {
                    "wavelength_m": wavelength_m,
                    "omega_rad_s": omega_by_wavelength[wavelength_m],
                    "x_m": float(x_value),
                    "x_over_L": float(x_value / BODY_LENGTH_M),
                    "response_curvature": float(response_curvature[index]),
                    "modal_curvature": float(modal_curvature[index]),
                    "density_indicator": float(density[index]),
                    "uniform_abs_error": float(uniform_error[index]),
                    "target_best_abs_error": float(best_error[index]),
                    "local_improvement": float(local_improvement[index]),
                    "target_best_layout": best_layout,
                }
            )

        lengths = layouts[best_layout]
        boundaries = module_boundaries(lengths)
        avg_density = module_average(x_m, density, boundaries)
        avg_response_curvature = module_average(x_m, response_curvature, boundaries)
        avg_modal_curvature = module_average(x_m, modal_curvature, boundaries)
        avg_improvement = module_average(x_m, local_improvement, boundaries)
        for module_id, (start, end, length) in enumerate(zip(boundaries[:-1], boundaries[1:], lengths), start=1):
            module_rows.append(
                {
                    "wavelength_m": wavelength_m,
                    "layout_id": best_layout,
                    "module_id": module_id,
                    "module_length_m": length,
                    "x_start_m": float(start),
                    "x_end_m": float(end),
                    "center_x_m": float(0.5 * (start + end)),
                    "avg_density_indicator": avg_density[module_id - 1],
                    "avg_response_curvature": avg_response_curvature[module_id - 1],
                    "avg_modal_curvature": avg_modal_curvature[module_id - 1],
                    "avg_local_improvement": avg_improvement[module_id - 1],
                }
            )

        short_density = [value for value, length in zip(avg_density, lengths) if np.isclose(length, 20.0)]
        long_density = [value for value, length in zip(avg_density, lengths) if np.isclose(length, 40.0)]
        validation_rows.append(
            {
                "wavelength_m": wavelength_m,
                "target_best_layout": best_layout,
                "density_vs_uniform_error_corr": safe_corr(density, uniform_error),
                "density_vs_local_improvement_corr": safe_corr(density, local_improvement),
                "short_module_mean_density": float(np.mean(short_density)) if short_density else float("nan"),
                "long_module_mean_density": float(np.mean(long_density)) if long_density else float("nan"),
                "short_minus_long_density": (
                    float(np.mean(short_density) - np.mean(long_density))
                    if short_density and long_density
                    else float("nan")
                ),
                "top_density_x_m": " ".join(f"{value:.1f}" for value in x_m[np.argsort(density)[-5:][::-1]]),
            }
        )

        for rank, mode_index in enumerate(np.argsort(participation)[-12:][::-1], start=1):
            modal_rows.append(
                {
                    "wavelength_m": wavelength_m,
                    "rank": rank,
                    "mode_index_one_based": int(mode_index + 1),
                    "structural_omega_rad_s": float(np.sqrt(max(eigenvalues[mode_index], 0.0))),
                    "wave_omega_rad_s": omega_by_wavelength[wavelength_m],
                    "participation_weight": float(participation[mode_index]),
                }
            )

    paths = {
        "density": write_csv(TABLE_DIR / "frequency_mode_density_by_x.csv", density_rows),
        "module": write_csv(TABLE_DIR / "density_by_target_best_module.csv", module_rows),
        "validation": write_csv(TABLE_DIR / "density_indicator_validation_summary.csv", validation_rows),
        "modal": write_csv(TABLE_DIR / "modal_participation_by_wavelength.csv", modal_rows),
    }
    return paths


def plot_density_panel(density_csv: Path, module_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(density_csv)
    module_rows = read_csv(module_csv)
    length_colors = {20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}

    fig, axes = plt.subplots(len(TARGET_WAVELENGTHS_M), 1, figsize=(11.8, 12.0), sharex=True)
    fig.suptitle("Frequency- and mode-adaptive control-point density indicator", fontsize=15)
    for axis, wavelength_m in zip(axes, TARGET_WAVELENGTHS_M):
        current = [row for row in rows if int(row["wavelength_m"]) == wavelength_m]
        modules = [row for row in module_rows if int(row["wavelength_m"]) == wavelength_m]
        x_m = np.asarray([float(row["x_m"]) for row in current])
        density = np.asarray([float(row["density_indicator"]) for row in current])
        response_curvature = np.asarray([float(row["response_curvature"]) for row in current])
        modal_curvature = np.asarray([float(row["modal_curvature"]) for row in current])
        improvement = normalize(np.asarray([float(row["local_improvement"]) for row in current]))

        for module in modules:
            start = float(module["x_start_m"])
            end = float(module["x_end_m"])
            length = float(module["module_length_m"])
            axis.axvspan(start, end, color=length_colors.get(length, "#dddddd"), alpha=0.13)
            axis.axvline(start, color="#333333", linewidth=0.4, alpha=0.55)
        axis.axvline(float(modules[-1]["x_end_m"]), color="#333333", linewidth=0.4, alpha=0.55)

        axis.plot(x_m, density, color="#111111", linewidth=2.0, label="density indicator")
        axis.plot(x_m, response_curvature, color="#1f77b4", linestyle="--", linewidth=1.2, label="response curvature")
        axis.plot(x_m, modal_curvature, color="#ff7f0e", linestyle=":", linewidth=1.8, label="mode-weighted curvature")
        axis.plot(x_m, improvement, color="#2ca02c", linewidth=1.0, alpha=0.7, label="normalized local improvement")
        centers = [float(module["center_x_m"]) for module in modules]
        axis.scatter(centers, [1.07] * len(centers), s=24, color="#111111", marker="v", clip_on=False)
        axis.set_ylim(-0.04, 1.12)
        axis.set_ylabel(f"{wavelength_m} m")
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
        axis.text(0.01, 0.83, modules[0]["layout_id"], transform=axis.transAxes, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xlabel("x along floating body (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = FIGURE_DIR / "frequency_mode_density_indicator_panel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_module_density(module_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(module_csv)
    length_colors = {20.0: "#66c2a5", 30.0: "#fc8d62", 40.0: "#8da0cb"}
    fig, axes = plt.subplots(len(TARGET_WAVELENGTHS_M), 1, figsize=(11.5, 10.4), sharex=True)
    fig.suptitle("Target-best module length versus average density demand", fontsize=15)
    for axis, wavelength_m in zip(axes, TARGET_WAVELENGTHS_M):
        current = [row for row in rows if int(row["wavelength_m"]) == wavelength_m]
        centers = np.asarray([float(row["center_x_m"]) for row in current])
        lengths = np.asarray([float(row["module_length_m"]) for row in current])
        density = np.asarray([float(row["avg_density_indicator"]) for row in current])
        axis.bar(
            centers,
            density,
            width=lengths * 0.86,
            color=[length_colors.get(length, "#dddddd") for length in lengths],
            edgecolor="#333333",
            linewidth=0.5,
        )
        for center, value, length in zip(centers, density, lengths):
            axis.text(center, value + 0.025, str(int(length)), ha="center", va="bottom", fontsize=8)
        axis.set_ylim(0.0, 1.16)
        axis.set_ylabel(f"{wavelength_m} m\navg D")
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
        axis.text(0.01, 0.82, current[0]["layout_id"], transform=axis.transAxes, fontsize=9)
    axes[-1].set_xlabel("x along floating body (m); bar label is module length (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIGURE_DIR / "density_vs_target_module_length_panel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_modal_participation(modal_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(modal_csv)
    top_modes = sorted({int(row["mode_index_one_based"]) for row in rows})
    values = np.zeros((len(top_modes), len(TARGET_WAVELENGTHS_M)))
    for row in rows:
        mode = int(row["mode_index_one_based"])
        wavelength = int(row["wavelength_m"])
        values[top_modes.index(mode), TARGET_WAVELENGTHS_M.index(wavelength)] = max(
            values[top_modes.index(mode), TARGET_WAVELENGTHS_M.index(wavelength)],
            float(row["participation_weight"]),
        )
    plot_values = np.log10(np.maximum(values, 1.0e-12))
    fig, axis = plt.subplots(figsize=(8.8, max(4.8, 0.22 * len(top_modes))))
    image = axis.imshow(plot_values, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(TARGET_WAVELENGTHS_M)))
    axis.set_xticklabels([str(value) for value in TARGET_WAVELENGTHS_M])
    axis.set_yticks(np.arange(len(top_modes)))
    axis.set_yticklabels([str(value) for value in top_modes])
    axis.set_xlabel("wavelength (m)")
    axis.set_ylabel("structural mode index")
    axis.set_title("Frequency-dependent modal participation weights")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("log10(participation weight)")
    fig.tight_layout()
    path = FIGURE_DIR / "modal_participation_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_density_error_relationship(density_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(density_csv)
    density = np.asarray([float(row["density_indicator"]) for row in rows])
    uniform_error = np.asarray([float(row["uniform_abs_error"]) for row in rows])
    improvement = np.asarray([float(row["local_improvement"]) for row in rows])
    wavelengths = np.asarray([int(row["wavelength_m"]) for row in rows])
    colors = {120: "#1f77b4", 180: "#9467bd", 240: "#ff7f0e", 300: "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for wavelength_m in TARGET_WAVELENGTHS_M:
        mask = wavelengths == wavelength_m
        axes[0].scatter(density[mask], uniform_error[mask], s=18, color=colors[wavelength_m], alpha=0.75, label=f"{wavelength_m} m")
        axes[1].scatter(density[mask], improvement[mask], s=18, color=colors[wavelength_m], alpha=0.75)
    axes[0].set_xlabel("density indicator")
    axes[0].set_ylabel("|U10 - U30|")
    axes[0].set_title("Density demand versus baseline error")
    axes[0].grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("density indicator")
    axes[1].set_ylabel("|U10 - U30| - |NU10 - U30|")
    axes[1].set_title("Density demand versus NU10 local improvement")
    axes[1].grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    fig.suptitle("Validation of the density indicator as an error-demand proxy", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURE_DIR / "density_error_relationship.png"
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


def write_report(paths: dict[str, Path], figures: dict[str, Path]) -> None:
    validation = read_csv(paths["validation"])
    validation_rows = [
        (
            row["wavelength_m"],
            row["target_best_layout"],
            f"{float(row['density_vs_uniform_error_corr']):.3f}",
            f"{float(row['density_vs_local_improvement_corr']):.3f}",
            f"{float(row['short_module_mean_density']):.3f}",
            f"{float(row['long_module_mean_density']):.3f}",
            row["top_density_x_m"],
        )
        for row in validation
    ]
    lines = [
        "# Frequency- and mode-adaptive 控制点密度指标",
        "",
        "## 1. 目的",
        "",
        "本步骤回答：为什么某些区域需要更多控制点？这里先构造一个诊断型密度指标，用 U30 参考响应解释控制点需求。后续做最少控制点选择算法时，可以把 U30 响应替换为低阶 pilot RODM 响应。",
        "",
        "## 2. 指标定义",
        "",
        "对目标频率 `omega`，定义控制点密度需求：",
        "",
        "$$",
        "D(x,\\omega)=\\eta\\,\\widehat{C_u}(x,\\omega)+(1-\\eta)\\sum_{r=1}^{R}p_r(\\omega)\\,\\widehat{C_{\\phi_r}}(x)",
        "$$",
        "",
        "其中 `C_u` 是目标频率下 U30 heave 响应沿长度方向的曲率强度，`C_phi_r` 是第 `r` 阶结构模态的垂向模态曲率强度，`p_r(omega)` 是由响应投影得到的频率相关模态参与权重。本文当前取 `eta=0.5`，使用前 80 阶结构模态。",
        "",
        "## 3. 密度指标与目标最优布局",
        "",
        f"![density panel]({file_uri(figures['density_panel'])})",
        "",
        "黑线是综合密度指标，蓝线是频率响应曲率，橙线是模态参与加权后的模态曲率，绿色线是归一化局部改善。背景颜色表示目标波长下最优 NU10 布局的模块长度。",
        "",
        f"![module density]({file_uri(figures['module_density'])})",
        "",
        "这个图把每个目标最优模块内的平均密度需求画成柱状图，并在柱顶标注模块长度。它用于判断短模块是否确实落在高密度需求区域。",
        "",
        markdown_table(
            (
                "wavelength (m)",
                "target-best layout",
                "corr D vs U10 error",
                "corr D vs local improvement",
                "20 m mean D",
                "40 m mean D",
                "top density x (m)",
            ),
            validation_rows,
        ),
        "",
        "## 4. 模态参与的频率适应性",
        "",
        f"![modal participation]({file_uri(figures['modal_participation'])})",
        "",
        "不同目标波长下，响应投影到结构模态上的权重不同。因此控制点需求不应该只由几何位置或单一曲率决定，而应随频率和主导模态变化。",
        "",
        "## 5. 与误差需求的关系",
        "",
        f"![density error]({file_uri(figures['density_error'])})",
        "",
        "密度指标与 U10 基线局部误差、NU10 局部改善之间并非一一线性关系，但它提供了一个可解释的空间需求场：高密度区域通常代表响应变化快、模态贡献强或基线误差更需要关注的位置。",
        "",
        "## 6. 当前结论",
        "",
        "1. 该指标把频率响应曲率和结构模态参与合并起来，比单纯看 U30 曲率更适合作为控制点需求解释。",
        "2. 不同波长下的高密度区域不同，支持“非均匀模块是目标频率相关策略”这一主线。",
        "3. 该指标目前是诊断型，使用 U30 参考响应；下一步可替换为低阶 pilot RODM 响应，并用于最少控制点选择算法。",
        "4. 节点对齐不是创新点，而是算法约束；真正的方法贡献应是 frequency- and mode-adaptive 的控制点需求指标以及后续的最少控制点选择。",
        "",
        "## 7. 输出文件",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{path}`")
    for key, path in figures.items():
        lines.append(f"- `figure_{key}`: `{path}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_m = centerline_x()
    eigenvalues, modes, mass = load_or_compute_modes(DEFAULT_MODE_COUNT, force=False)
    modal_curvatures = modal_centerline_curvatures(modes, x_m)
    paths = build_density_tables(
        eigenvalues=eigenvalues,
        modes=modes,
        mass=mass,
        modal_curvatures=modal_curvatures,
        x_m=x_m,
    )
    figures = {
        "density_panel": plot_density_panel(paths["density"], paths["module"]),
        "module_density": plot_module_density(paths["module"]),
        "modal_participation": plot_modal_participation(paths["modal"]),
        "density_error": plot_density_error_relationship(paths["density"]),
    }
    write_report(paths, figures)
    manifest = OUTPUT_ROOT / "frequency_mode_adaptive_density_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "tables": {key: str(value) for key, value in paths.items()},
                "figures": {key: str(value) for key, value in figures.items()},
                "modal_cache": str(MODAL_CACHE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    print(f"tables={TABLE_DIR}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
