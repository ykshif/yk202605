"""Compare Newmark and RK4 time integration in the external WEC-Sim-like adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain import relative_l2_error, zero_mean_rms  # noqa: E402
from run_time_domain_reference_case_300 import centerline_heave_time  # noqa: E402
from run_wecsim_like_time_domain_platform import (  # noqa: E402
    DEFAULT_HYDRO_FILE,
    solve_selected_models,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "time_integrator_newmark_rk4_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=0.4157)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--excitation-model", choices=("regular_wave", "wave_spectrum"), default="wave_spectrum")
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-type", choices=("jonswap", "pierson_moskowitz"), default="jonswap")
    parser.add_argument("--peak-enhancement-factor", type=float, default=3.3)
    parser.add_argument("--spectrum-seed", type=int, default=1)
    parser.add_argument("--wave-amplitude", type=float, default=1.0)
    parser.add_argument("--phase-rad", type=float, default=0.0)
    parser.add_argument("--ramp-cycles", type=float, default=5.0)
    parser.add_argument("--cycles", type=float, default=20.0)
    parser.add_argument("--steps-per-cycle", type=int, default=80)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
    parser.add_argument("--state-order", type=int, default=240)
    parser.add_argument("--era-block-rows", type=int, default=55)
    parser.add_argument("--era-block-cols", type=int, default=55)
    parser.add_argument("--radiation-passivity-correction", choices=("none", "clip_negative_eigenvalues"), default="clip_negative_eigenvalues")
    parser.add_argument("--radiation-convolution-rule", choices=("rectangular", "trapezoidal"), default="trapezoidal")
    parser.add_argument("--radiation-residual-model", choices=("none", "selected_frequency"), default="selected_frequency")
    parser.add_argument("--mooring-grid-nodes-x", type=int, default=61)
    parser.add_argument("--mooring-grid-nodes-y", type=int, default=13)
    parser.add_argument("--mooring-corner-horizontal-stiffness", type=float, default=1.0e7)
    parser.add_argument("--mooring-corner-surge-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-sway-stiffness", type=float, default=None)
    parser.add_argument("--mooring-corner-heave-stiffness", type=float, default=0.0)
    return parser.parse_args()


def relative_metrics(candidate, reference) -> dict[str, float]:
    candidate_heave = centerline_heave_time(candidate.global_displacement, retained_dofs_per_node=5)
    reference_heave = centerline_heave_time(reference.global_displacement, retained_dofs_per_node=5)
    return {
        "master_displacement_l2_relative_error": relative_l2_error(
            candidate.master_displacement,
            reference.master_displacement,
        ),
        "master_velocity_l2_relative_error": relative_l2_error(
            candidate.master_velocity,
            reference.master_velocity,
        ),
        "memory_force_l2_relative_error": relative_l2_error(
            candidate.memory_force,
            reference.memory_force,
        ),
        "global_displacement_l2_relative_error": relative_l2_error(
            candidate.global_displacement,
            reference.global_displacement,
        ),
        "centerline_heave_l2_relative_error": relative_l2_error(
            candidate_heave,
            reference_heave,
        ),
        "centerline_heave_rms_relative_error": relative_l2_error(
            zero_mean_rms(candidate_heave, axis=0),
            zero_mean_rms(reference_heave, axis=0),
        ),
        "reference_centerline_heave_rms_mean": float(np.mean(zero_mean_rms(reference_heave, axis=0))),
        "candidate_centerline_heave_rms_mean": float(np.mean(zero_mean_rms(candidate_heave, axis=0))),
    }


def plot_heave_rms(path: Path, newmark_results: dict, rk4_results: dict) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, model_name in zip(axes, ("direct_convolution", "state_space")):
        newmark_heave = centerline_heave_time(
            newmark_results[model_name].global_displacement,
            retained_dofs_per_node=5,
        )
        rk4_heave = centerline_heave_time(
            rk4_results[model_name].global_displacement,
            retained_dofs_per_node=5,
        )
        x = np.linspace(0.0, 1.0, newmark_heave.shape[1])
        ax.plot(x, zero_mean_rms(newmark_heave, axis=0), color="#111111", linewidth=1.4, label="Newmark")
        ax.plot(x, zero_mean_rms(rk4_heave, axis=0), color="#d62728", linestyle="--", linewidth=1.2, label="RK4")
        ax.set_xlabel("x/L")
        ax.set_title(model_name)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Centerline heave RMS")
    axes[0].legend(frameon=False)
    fig.suptitle("Reduced-space time integrator comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def _representative_columns(column_count: int) -> tuple[int, int, int]:
    if column_count < 3:
        raise ValueError("at least three centerline samples are required")
    return 0, column_count // 2, column_count - 1


def plot_representative_heave_time(
    path: Path,
    newmark_results: dict,
    rk4_results: dict,
    model_name: str,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    newmark_result = newmark_results[model_name]
    rk4_result = rk4_results[model_name]
    time = np.asarray(newmark_result.time, dtype=float)
    if not np.allclose(time, rk4_result.time):
        raise ValueError("Newmark and RK4 time grids must match for plotting")

    newmark_heave = centerline_heave_time(
        newmark_result.global_displacement,
        retained_dofs_per_node=5,
    )
    rk4_heave = centerline_heave_time(
        rk4_result.global_displacement,
        retained_dofs_per_node=5,
    )
    columns = _representative_columns(newmark_heave.shape[1])
    labels = ("x/L=0.0", "x/L=0.5", "x/L=1.0")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, column, label in zip(axes, columns, labels):
        ax.plot(time, newmark_heave[:, column], color="#111111", linewidth=1.2, label="Newmark")
        ax.plot(time, rk4_heave[:, column], color="#d62728", linestyle="--", linewidth=1.0, label="RK4")
        ax.set_ylabel("heave")
        ax.set_title(label)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False, ncols=2)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{model_name}: representative centerline heave time histories")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_error_norm_time(path: Path, newmark_results: dict, rk4_results: dict) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    for ax, model_name in zip(axes, ("direct_convolution", "state_space")):
        newmark_result = newmark_results[model_name]
        rk4_result = rk4_results[model_name]
        time = np.asarray(newmark_result.time, dtype=float)
        if not np.allclose(time, rk4_result.time):
            raise ValueError("Newmark and RK4 time grids must match for plotting")
        reference_norm = np.linalg.norm(newmark_result.master_displacement, axis=1)
        error_norm = np.linalg.norm(
            rk4_result.master_displacement - newmark_result.master_displacement,
            axis=1,
        )
        scale = max(float(np.max(reference_norm)), np.finfo(float).eps)
        ax.plot(time, error_norm / scale, color="#1f77b4", linewidth=1.2)
        ax.set_ylabel("relative norm")
        ax.set_title(model_name)
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("RK4 minus Newmark master-displacement error history")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def run_for_integrator(args: argparse.Namespace, integrator: str):
    run_args = argparse.Namespace(**vars(args))
    run_args.output_root = args.output_root / integrator
    run_args.radiation_model = "both"
    run_args.integrator = integrator
    run_args.save_state_space_model_path = None
    run_args.state_space_model_path = None
    run_args.save_arrays = False
    return solve_selected_models(run_args)


def main() -> int:
    args = parse_args()
    start = timer.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)

    newmark_results, case_metadata, _ = run_for_integrator(args, "newmark")
    rk4_results, _, _ = run_for_integrator(args, "rk4")
    comparisons = {
        model_name: relative_metrics(rk4_results[model_name], newmark_results[model_name])
        for model_name in ("direct_convolution", "state_space")
    }
    figures = [
        plot_heave_rms(
            args.output_root / "figures" / "newmark_vs_rk4_centerline_heave_rms.png",
            newmark_results,
            rk4_results,
        )
    ]
    figures.extend(
        [
            plot_representative_heave_time(
                args.output_root / "figures" / "direct_convolution_newmark_vs_rk4_heave_time.png",
                newmark_results,
                rk4_results,
                "direct_convolution",
            ),
            plot_representative_heave_time(
                args.output_root / "figures" / "state_space_newmark_vs_rk4_heave_time.png",
                newmark_results,
                rk4_results,
                "state_space",
            ),
            plot_error_norm_time(
                args.output_root / "figures" / "newmark_vs_rk4_error_norm_time.png",
                newmark_results,
                rk4_results,
            ),
        ]
    )
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "case": case_metadata,
        "settings": {
            "cycles": args.cycles,
            "steps_per_cycle": args.steps_per_cycle,
            "memory_cycles": args.memory_cycles,
            "state_order": args.state_order,
            "era_block_rows": args.era_block_rows,
            "era_block_cols": args.era_block_cols,
            "mooring_corner_horizontal_stiffness": args.mooring_corner_horizontal_stiffness,
        },
        "comparison_reference": "newmark",
        "comparison_candidate": "rk4",
        "comparisons": comparisons,
        "figures": figures,
        "elapsed_seconds": float(timer.perf_counter() - start),
    }
    metrics_path = write_metrics_json(args.output_root / "time_integrator_comparison_metrics.json", metrics)

    print("Time-integrator comparison completed.")
    for model_name, values in comparisons.items():
        print(
            f"{model_name}: master={values['master_displacement_l2_relative_error']:.6g}, "
            f"memory={values['memory_force_l2_relative_error']:.6g}, "
            f"heave_rms={values['centerline_heave_rms_relative_error']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
