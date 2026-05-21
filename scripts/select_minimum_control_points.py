"""Select the minimum number of control points from the adaptive density field.

This script implements the first algorithmic step after the
frequency-/mode-adaptive density indicator:

    minimize number of 1D hydrodynamic modules / control points
    subject to a density-weighted surrogate error tolerance.

The selected layouts are still module partitions along the length direction
only. Module lengths are restricted to 10 m multiples so every module center
falls on the 5 m FEM centerline grid.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))


DENSITY_ROOT = REPO_ROOT / "results" / "frequency_mode_adaptive_density"
DENSITY_CSV = DENSITY_ROOT / "tables" / "frequency_mode_density_by_x.csv"

OUTPUT_ROOT = REPO_ROOT / "results" / "minimum_control_point_selection"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = OUTPUT_ROOT / "minimum_control_point_selection_report.md"

BODY_LENGTH_M = 300.0
STRUCTURAL_DX_M = 5.0
STRUCTURAL_DY_M = 5.0
BODY_WIDTH_M = 60.0
STRUCTURAL_NODES_PER_X = int(round(BODY_LENGTH_M / STRUCTURAL_DX_M)) + 1
STRUCTURAL_CENTERLINE_Y_INDEX = int(round((BODY_WIDTH_M / 2.0) / STRUCTURAL_DY_M))
STRUCTURAL_CENTERLINE_Y_M = BODY_WIDTH_M / 2.0

BOUNDARY_STEP_M = 10.0
ALLOWED_LENGTHS_M = (10, 20, 30, 40, 50, 60)
MIN_MODULES = 5
MAX_MODULES = 20
TOLERANCE_RATIOS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
DENSITY_FLOOR = 0.08
TARGET_WAVELENGTHS_M = (120, 180, 240, 300)


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
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    span = float(np.max(values) - np.min(values))
    if span <= 1.0e-14:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def nearest_centerline_node(x_m: float) -> tuple[int, float, float]:
    x_index = int(math.floor(x_m / STRUCTURAL_DX_M + 0.5))
    x_index = min(max(x_index, 0), STRUCTURAL_NODES_PER_X - 1)
    selected_x = x_index * STRUCTURAL_DX_M
    node_x_index = STRUCTURAL_NODES_PER_X - 1 - x_index
    node_id = STRUCTURAL_CENTERLINE_Y_INDEX * STRUCTURAL_NODES_PER_X + node_x_index + 1
    return node_id, selected_x, STRUCTURAL_CENTERLINE_Y_M


def load_density_by_target() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = read_csv(DENSITY_CSV)
    x_values = sorted({float(row["x_m"]) for row in rows})
    x_m = np.asarray(x_values)
    by_wavelength: dict[int, np.ndarray] = {}
    for wavelength_m in TARGET_WAVELENGTHS_M:
        current = [row for row in rows if int(row["wavelength_m"]) == wavelength_m]
        current.sort(key=lambda row: float(row["x_m"]))
        by_wavelength[wavelength_m] = normalize(np.asarray([float(row["density_indicator"]) for row in current]))

    targets: dict[str, np.ndarray] = {
        f"wl_{wavelength_m}m": by_wavelength[wavelength_m] for wavelength_m in TARGET_WAVELENGTHS_M
    }
    targets["band_equal_120_300m"] = normalize(
        np.mean([by_wavelength[wavelength_m] for wavelength_m in TARGET_WAVELENGTHS_M], axis=0)
    )
    targets["band_center_120_240m"] = normalize(
        np.mean([by_wavelength[120], by_wavelength[240]], axis=0)
    )
    return x_m, targets


def segment_cost(
    x_m: np.ndarray,
    density: np.ndarray,
    start_m: float,
    end_m: float,
) -> float:
    dense_x = np.linspace(start_m, end_m, max(5, int(round(end_m - start_m)) + 1))
    dense_density = np.interp(dense_x, x_m, density)
    center = 0.5 * (start_m + end_m)
    weights = DENSITY_FLOOR + dense_density
    moment = np.trapz(weights * (dense_x - center) ** 2, dense_x)
    return float(moment / BODY_LENGTH_M**3)


def layout_cost(
    x_m: np.ndarray,
    density: np.ndarray,
    lengths_m: tuple[int, ...],
) -> float:
    start = 0.0
    total = 0.0
    for length in lengths_m:
        end = start + float(length)
        total += segment_cost(x_m, density, start, end)
        start = end
    if not np.isclose(start, BODY_LENGTH_M):
        raise ValueError(f"layout length sum is {start}, expected {BODY_LENGTH_M}")
    return total


def optimize_for_module_count(
    x_m: np.ndarray,
    density: np.ndarray,
    module_count: int,
) -> tuple[float, tuple[int, ...]]:
    positions = np.arange(0.0, BODY_LENGTH_M + BOUNDARY_STEP_M, BOUNDARY_STEP_M)
    position_count = len(positions)
    allowed_steps = tuple(int(length / BOUNDARY_STEP_M) for length in ALLOWED_LENGTHS_M)
    cost_cache: dict[tuple[int, int], float] = {}
    for end_index in range(1, position_count):
        for step in allowed_steps:
            start_index = end_index - step
            if start_index < 0:
                continue
            cost_cache[(start_index, end_index)] = segment_cost(
                x_m,
                density,
                positions[start_index],
                positions[end_index],
            )

    inf = float("inf")
    dp = np.full((module_count + 1, position_count), inf)
    previous = np.full((module_count + 1, position_count), -1, dtype=int)
    dp[0, 0] = 0.0
    for used in range(1, module_count + 1):
        for end_index in range(1, position_count):
            best_cost = inf
            best_start = -1
            for step in allowed_steps:
                start_index = end_index - step
                if start_index < 0:
                    continue
                candidate = dp[used - 1, start_index] + cost_cache[(start_index, end_index)]
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start_index
            dp[used, end_index] = best_cost
            previous[used, end_index] = best_start

    final_index = position_count - 1
    if not np.isfinite(dp[module_count, final_index]):
        return inf, ()

    lengths = []
    current = final_index
    for used in range(module_count, 0, -1):
        start = previous[used, current]
        if start < 0:
            return inf, ()
        lengths.append(int(round((positions[current] - positions[start]))))
        current = start
    lengths.reverse()
    return float(dp[module_count, final_index]), tuple(lengths)


def module_centers(lengths_m: tuple[int, ...]) -> np.ndarray:
    boundaries = np.concatenate([[0.0], np.cumsum(np.asarray(lengths_m, dtype=float))])
    return 0.5 * (boundaries[:-1] + boundaries[1:])


def write_algorithm_outputs(x_m: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, Path]:
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []

    uniform_lengths = (30,) * 10
    for target_id, density in targets.items():
        uniform_cost = layout_cost(x_m, density, uniform_lengths)
        target_candidates = []
        for module_count in range(MIN_MODULES, MAX_MODULES + 1):
            cost, lengths = optimize_for_module_count(x_m, density, module_count)
            if not lengths:
                continue
            ratio = cost / uniform_cost if uniform_cost > 0.0 else float("nan")
            centers = module_centers(lengths)
            node_ids = [nearest_centerline_node(center)[0] for center in centers]
            row = {
                "target_id": target_id,
                "module_count": module_count,
                "surrogate_cost": cost,
                "cost_ratio_vs_U10_uniform": ratio,
                "module_lengths_m": " ".join(str(length) for length in lengths),
                "module_centers_m": " ".join(f"{center:.1f}" for center in centers),
                "selected_node_ids": " ".join(str(node_id) for node_id in node_ids),
                "max_module_length_m": max(lengths),
                "min_module_length_m": min(lengths),
            }
            candidate_rows.append(row)
            target_candidates.append(row)

        for tolerance in TOLERANCE_RATIOS:
            feasible = [
                row
                for row in target_candidates
                if float(row["cost_ratio_vs_U10_uniform"]) <= tolerance
            ]
            if not feasible:
                continue
            selected = min(
                feasible,
                key=lambda row: (int(row["module_count"]), float(row["cost_ratio_vs_U10_uniform"])),
            )
            selection_id = f"{target_id}_tol{tolerance:g}_N{selected['module_count']}"
            selected_row = dict(selected)
            selected_row["selection_id"] = selection_id
            selected_row["tolerance_ratio_vs_U10_uniform"] = tolerance
            selected_rows.append(selected_row)

            lengths = tuple(int(value) for value in str(selected["module_lengths_m"]).split())
            boundaries = np.concatenate([[0.0], np.cumsum(np.asarray(lengths, dtype=float))])
            centers = module_centers(lengths)
            for module_id, (start, end, center, length) in enumerate(
                zip(boundaries[:-1], boundaries[1:], centers, lengths),
                start=1,
            ):
                node_id, node_x, node_y = nearest_centerline_node(float(center))
                geometry_rows.append(
                    {
                        "selection_id": selection_id,
                        "target_id": target_id,
                        "tolerance_ratio_vs_U10_uniform": tolerance,
                        "module_count": selected["module_count"],
                        "module_id": module_id,
                        "module_length_m": length,
                        "x_start_m": float(start),
                        "x_end_m": float(end),
                        "center_x_m": float(center),
                        "selected_node_id": node_id,
                        "selected_node_x_m": node_x,
                        "selected_node_y_m": node_y,
                        "abs_node_error_m": abs(node_x - center),
                    }
                )

    paths = {
        "candidates": write_csv(TABLE_DIR / "minimum_control_point_candidate_layouts.csv", candidate_rows),
        "selected": write_csv(TABLE_DIR / "minimum_control_points_by_tolerance.csv", selected_rows),
        "geometry": write_csv(TABLE_DIR / "selected_minimum_control_point_geometry.csv", geometry_rows),
    }
    return paths


def plot_cost_curves(candidate_csv: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = read_csv(candidate_csv)
    target_ids = sorted({row["target_id"] for row in rows})
    fig, axis = plt.subplots(figsize=(11.2, 6.0))
    for target_id in target_ids:
        current = [row for row in rows if row["target_id"] == target_id]
        current.sort(key=lambda row: int(row["module_count"]))
        axis.plot(
            [int(row["module_count"]) for row in current],
            [float(row["cost_ratio_vs_U10_uniform"]) for row in current],
            marker="o",
            linewidth=1.6,
            label=target_id,
        )
    for tolerance in (1.0, 0.8, 0.6, 0.5):
        axis.axhline(tolerance, color="#333333", linewidth=0.7, linestyle="--", alpha=0.45)
        axis.text(MAX_MODULES + 0.15, tolerance, f"{tolerance:g}", va="center", fontsize=8)
    axis.set_xlabel("number of control points / modules")
    axis.set_ylabel("density-weighted surrogate cost / U10 cost")
    axis.set_title("Minimum control-point search curves")
    axis.set_xticks(range(MIN_MODULES, MAX_MODULES + 1))
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIGURE_DIR / "minimum_control_point_cost_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_selected_layouts(
    x_m: np.ndarray,
    targets: dict[str, np.ndarray],
    selected_csv: Path,
    *,
    tolerance: float = 0.8,
) -> Path:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in read_csv(selected_csv)
        if np.isclose(float(row["tolerance_ratio_vs_U10_uniform"]), tolerance)
    ]
    rows.sort(key=lambda row: row["target_id"])
    length_colors = {
        10: "#8dd3c7",
        20: "#66c2a5",
        30: "#fc8d62",
        40: "#8da0cb",
        50: "#e78ac3",
        60: "#a6d854",
    }

    fig, axes = plt.subplots(len(rows), 1, figsize=(11.5, 3.1 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    fig.suptitle(f"Selected minimum layouts at surrogate tolerance {tolerance:g}", fontsize=15)
    for axis, row in zip(axes, rows):
        target_id = row["target_id"]
        density = targets[target_id]
        lengths = tuple(int(value) for value in row["module_lengths_m"].split())
        boundaries = np.concatenate([[0.0], np.cumsum(np.asarray(lengths, dtype=float))])
        centers = module_centers(lengths)
        for start, end, length in zip(boundaries[:-1], boundaries[1:], lengths):
            axis.axvspan(start, end, color=length_colors[length], alpha=0.16)
            axis.axvline(start, color="#333333", linewidth=0.45, alpha=0.55)
        axis.axvline(boundaries[-1], color="#333333", linewidth=0.45, alpha=0.55)
        axis.plot(x_m, density, color="#111111", linewidth=2.0, label="density indicator")
        axis.scatter(centers, [1.05] * len(centers), marker="v", s=28, color="#111111", clip_on=False)
        for center, length in zip(centers, lengths):
            axis.text(center, 0.06, str(length), ha="center", va="bottom", fontsize=8)
        axis.set_ylim(-0.05, 1.12)
        axis.set_ylabel(target_id)
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
        axis.text(
            0.01,
            0.82,
            f"N={row['module_count']}, cost ratio={float(row['cost_ratio_vs_U10_uniform']):.3f}",
            transform=axis.transAxes,
            fontsize=9,
        )
    axes[-1].set_xlabel("x along floating body (m); numbers are module lengths (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIGURE_DIR / f"selected_minimum_layouts_tol{tolerance:g}.png"
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
    selected = read_csv(paths["selected"])
    key_rows = [
        row
        for row in selected
        if float(row["tolerance_ratio_vs_U10_uniform"]) in {1.0, 0.8, 0.6, 0.5}
    ]
    key_rows.sort(key=lambda row: (row["target_id"], -float(row["tolerance_ratio_vs_U10_uniform"])))
    table_rows = [
        (
            row["target_id"],
            row["tolerance_ratio_vs_U10_uniform"],
            row["module_count"],
            f"{float(row['cost_ratio_vs_U10_uniform']):.3f}",
            f"`[{', '.join(row['module_lengths_m'].split())}]`",
        )
        for row in key_rows
    ]
    lines = [
        "# 最少控制点选择算法：密度指标驱动的动态规划原型",
        "",
        "## 1. 目的",
        "",
        "本步骤回答：在给定误差容限下，如何选择最少控制点？当前实现使用 frequency- and mode-adaptive 密度指标作为低成本代理误差场，并用动态规划在一维模块划分中寻找最少模块数。",
        "",
        "## 2. 优化问题",
        "",
        "模块仍然沿长度方向一维划分。模块长度限制为 `10, 20, 30, 40, 50, 60 m`，因此所有模块重心都落在 5 m FEM 中线节点上。",
        "",
        "对给定目标密度 `D(x)`，每个模块 `[a,b]` 的代理误差定义为：",
        "",
        "$$",
        "E_{seg}=\\int_a^b (D(x)+D_0)(x-c)^2 dx, \\quad c=(a+b)/2",
        "$$",
        "",
        "动态规划在所有满足总长 300 m 的离散模块组合中最小化 `sum(E_seg)`。然后用 U10 uniform 的代理误差归一化，得到 `cost ratio`。在给定容限下，选择满足 `cost ratio <= tolerance` 的最小模块数。",
        "",
        "## 3. 成本曲线",
        "",
        f"![cost curves]({file_uri(figures['cost_curves'])})",
        "",
        "该图显示随着控制点数量增加，密度加权代理误差如何下降。横向虚线是不同误差容限。",
        "",
        "## 4. 代表性最少布局",
        "",
        f"![selected layouts]({file_uri(figures['selected_tol_0p8'])})",
        "",
        "图中黑线为目标密度指标，背景色为算法选择的模块区间，三角符号为控制点/模块重心位置，数字为模块长度。",
        "",
        markdown_table(
            ("target", "tolerance", "N", "cost ratio", "module lengths (m)"),
            table_rows,
        ),
        "",
        "## 5. 当前解释",
        "",
        "1. 这是最少控制点算法的第一版，目标是先把密度指标转化为自动布局。",
        "2. 当前误差是密度加权代理误差，不是最终 RODM 误差；它用于筛选少量候选布局。",
        "3. 下一步应对候选布局运行 Capytaine + SEREP-ridge RODM，验证真实 heave/结构响应误差是否满足容限。",
        "4. 若真实误差验证通过，这个算法就可以写成论文中的最少控制点选择方法；若不通过，则需要把真实 RODM 误差反馈进目标函数。",
        "",
        "## 6. 输出文件",
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
    x_m, targets = load_density_by_target()
    paths = write_algorithm_outputs(x_m, targets)
    figures = {
        "cost_curves": plot_cost_curves(paths["candidates"]),
        "selected_tol_0p8": plot_selected_layouts(x_m, targets, paths["selected"], tolerance=0.8),
        "selected_tol_0p6": plot_selected_layouts(x_m, targets, paths["selected"], tolerance=0.6),
    }
    write_report(paths, figures)
    manifest = OUTPUT_ROOT / "minimum_control_point_selection_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "report": str(REPORT_PATH),
                "tables": {key: str(value) for key, value in paths.items()},
                "figures": {key: str(value) for key, value in figures.items()},
                "allowed_lengths_m": ALLOWED_LENGTHS_M,
                "tolerance_ratios": TOLERANCE_RATIOS,
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
