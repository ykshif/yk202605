"""Compare 10x10 centerline responses for pitch hinge stiffness values."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.validation import (  # noqa: E402
    build_complex_hinge_10x10_case,
    missing_complex_hinge_input_paths,
    solve_complex_hinge_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "complex_hinge_10x10_pitch_stiffness_sweep"
DEFAULT_FIXED_COUPLING_STIFFNESS = 1.0e10
DEFAULT_PITCH_STIFFNESSES = (0.0, 1.0e8, 1.0e9, 1.0e10)


def _format_stiffness_label(value: float) -> str:
    """Return a compact label for figure legends and file names."""

    if value == 0:
        return "0"
    exponent_label = f"{value:.0e}".replace("+", "")
    return exponent_label.replace("e0", "e")


def _case_output_name(pitch_stiffness: float) -> str:
    """Return a filesystem-safe case suffix for one stiffness value."""

    return f"k_pitch{_format_stiffness_label(pitch_stiffness).replace('-', 'm')}"


def plot_centerline_comparison(
    records: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    """Plot center row and center column responses for all stiffness values."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "complex_hinge_10x10_centerline_pitch_stiffness_comparison.png"

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=False)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for color, record in zip(colors, records):
        label = f"k_pitch = {record['pitch_stiffness_label']}"
        x_coordinates = np.asarray(record["x_coordinates"])
        y_coordinates = np.asarray(record["y_coordinates"])
        center_row = np.asarray(record["center_row"])
        center_column = np.asarray(record["center_column"])
        axes[0].plot(x_coordinates, center_row, color=color, linewidth=1.8, label=label)
        axes[1].plot(y_coordinates, center_column, color=color, linewidth=1.8, label=label)

    axes[0].set_title("10x10 pitch hinge stiffness sweep: center row")
    axes[0].set_xlabel("x (m), y = 150 m")
    axes[0].set_ylabel("Heave displacement")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_title("10x10 pitch hinge stiffness sweep: center column")
    axes[1].set_xlabel("y (m), x = 150 m")
    axes[1].set_ylabel("Heave displacement")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.7)
    axes[1].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    figure_paths = [figure_path]

    high_k_records = [record for record in records if record["pitch_stiffness"] != 0.0]
    if len(high_k_records) >= 2:
        zoom_path = output_dir / "complex_hinge_10x10_centerline_pitch_stiffness_high_k_zoom.png"
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=False)
        for color, record in zip(colors[1:], high_k_records):
            label = f"k_pitch = {record['pitch_stiffness_label']}"
            x_coordinates = np.asarray(record["x_coordinates"])
            y_coordinates = np.asarray(record["y_coordinates"])
            center_row = np.asarray(record["center_row"])
            center_column = np.asarray(record["center_column"])
            axes[0].plot(x_coordinates, center_row, color=color, linewidth=1.8, label=label)
            axes[1].plot(y_coordinates, center_column, color=color, linewidth=1.8, label=label)

        axes[0].set_title("10x10 pitch hinge stiffness sweep: center row, high-k zoom")
        axes[0].set_xlabel("x (m), y = 150 m")
        axes[0].set_ylabel("Heave displacement")
        axes[0].grid(True, color="#d9d9d9", linewidth=0.7)
        axes[0].legend(frameon=False, ncol=3)

        axes[1].set_title("10x10 pitch hinge stiffness sweep: center column, high-k zoom")
        axes[1].set_xlabel("y (m), x = 150 m")
        axes[1].set_ylabel("Heave displacement")
        axes[1].grid(True, color="#d9d9d9", linewidth=0.7)
        axes[1].legend(frameon=False, ncol=3)

        fig.tight_layout()
        fig.savefig(zoom_path, dpi=300)
        plt.close(fig)
        figure_paths.append(zoom_path)

    return figure_paths


def write_report(output_root: Path, metrics: dict[str, object]) -> Path:
    """Write a short Chinese report for the stiffness sweep."""

    report_path = output_root / "report.md"
    lines = [
        "# 10x10 pitch 铰接刚度中心线响应对比报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 算例说明",
        "",
        "- 对象：10x10 模块化铰接浮体。",
        "- 水动力：`DM10_10_direction0_wl180.nc`，波长 180 m，入射方向 0 度。",
        f"- 固定参数：其他连接自由度刚度 `k_hinge = {metrics['fixed_coupling_stiffness_label']}`。",
        "- 扫描参数：pitch 铰接转动刚度，即旧程序铰接矩阵中原来取 `10` 的释放转动项。",
        "- 说明：刚度增大时，该转动释放项逐步接近其他自由度刚度，运动应趋向连续。",
        "- 对比位置：合并重复边界后的 61x61 位移云图中心行和中心列。",
        "",
        "## 2. 刚度列表",
        "",
    ]
    for item in metrics["records"]:
        lines.append(f"- `k_pitch = {item['pitch_stiffness_label']}`")

    lines.extend(
        [
            "",
            "## 3. 输出文件",
            "",
            "- 中心线对比图：",
        ]
    )
    for figure in metrics["figures"]:
        lines.append(f"- `{figure}`")
    lines.extend(
        [
            f"- 结果数据：`{metrics['records_path']}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="DM-FEM2D data root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    parser.add_argument(
        "--pitch-stiffness",
        nargs="*",
        type=float,
        default=list(DEFAULT_PITCH_STIFFNESSES),
        help="Pitch hinge stiffness values to compare",
    )
    parser.add_argument(
        "--fixed-coupling-stiffness",
        type=float,
        default=DEFAULT_FIXED_COUPLING_STIFFNESS,
        help="Stiffness kept unchanged for the other hinge connector DOFs",
    )
    return parser.parse_args()


def main() -> int:
    """Run the stiffness sweep and write comparison outputs."""

    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for pitch_stiffness in args.pitch_stiffness:
        case = build_complex_hinge_10x10_case(
            args.data_root,
            k_hinge=args.fixed_coupling_stiffness,
            released_dof_stiffness=pitch_stiffness,
        )
        missing = missing_complex_hinge_input_paths(case)
        if missing:
            raise FileNotFoundError(f"Missing inputs for {case.case_id}: {missing}")

        print(
            "Solving 10x10 complex hinge case with "
            f"k_pitch={pitch_stiffness:g}, "
            f"other DOFs k={args.fixed_coupling_stiffness:g} ..."
        )
        result = solve_complex_hinge_case(case)
        grid = result.heave_grid_merged
        center_row_index = grid.shape[0] // 2
        center_column_index = grid.shape[1] // 2
        x_coordinates = np.linspace(0.0, case.grid.structure_size, grid.shape[1])
        y_coordinates = np.linspace(0.0, case.grid.structure_size, grid.shape[0])

        suffix = _case_output_name(pitch_stiffness)
        grid_path = output_root / f"heave_grid_merged_{suffix}.npy"
        response_path = output_root / f"response_{suffix}.npy"
        np.save(grid_path, grid)
        np.save(response_path, result.response)

        records.append(
            {
                "fixed_coupling_stiffness": args.fixed_coupling_stiffness,
                "fixed_coupling_stiffness_label": _format_stiffness_label(
                    args.fixed_coupling_stiffness
                ),
                "pitch_stiffness": pitch_stiffness,
                "pitch_stiffness_label": _format_stiffness_label(pitch_stiffness),
                "omega": result.omega,
                "grid_shape": grid.shape,
                "grid_min": float(np.min(grid)),
                "grid_max": float(np.max(grid)),
                "grid_mean": float(np.mean(grid)),
                "center_row_index": center_row_index,
                "center_column_index": center_column_index,
                "x_coordinates": x_coordinates.tolist(),
                "y_coordinates": y_coordinates.tolist(),
                "center_row": grid[center_row_index, :].tolist(),
                "center_column": grid[:, center_column_index].tolist(),
                "grid_path": grid_path,
                "response_path": response_path,
            }
        )

    figure_paths = plot_centerline_comparison(records, output_root / "figures")
    records_path = output_root / "centerline_records.json"
    metrics_path = output_root / "metrics.json"
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "solved",
        "note": "已完成 10x10 pitch 铰接刚度中心线响应扫描。",
        "fixed_coupling_stiffness": args.fixed_coupling_stiffness,
        "fixed_coupling_stiffness_label": _format_stiffness_label(args.fixed_coupling_stiffness),
        "figures": figure_paths,
        "records_path": records_path,
        "records": records,
    }
    write_metrics_json(records_path, {"records": records})
    write_metrics_json(metrics_path, metrics)
    report_path = write_report(output_root, metrics)

    print("10x10 pitch hinge stiffness comparison completed.")
    for figure_path in figure_paths:
        print(f"Wrote {figure_path}")
    print(f"Wrote {records_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
