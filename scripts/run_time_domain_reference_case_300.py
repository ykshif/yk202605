"""Run the basic 300 m x 60 m RODM time-series case.

This is the user-facing entry point for the first time-domain workflow. It
uses the existing 10 hydrodynamic-node RODM case, writes full retained-DOF time
histories, extracts centerline heave histories, and compares the fitted
steady-state time-domain amplitude against the existing frequency-domain
solver unless disabled.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import csv
import os
import sys
import time as timer

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import (  # noqa: E402
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
    build_rodm_frequency_case_from_config,
    write_metrics_json,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    fit_harmonic_amplitude,
    harmonic_amplitude_error,
    solve_rodm_time_domain_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "reference_case_300_timeseries"
REPO_LOCAL_DATA_ROOT = REPO_ROOT / "data" / "external" / "DM-FEM2D"
CENTERLINE_START_NODE = 367
CENTERLINE_STOP_NODE_EXCLUSIVE = 427
HEAVE_DOF_ZERO_BASED = 2


def default_dm_fem_root(data_root: str | Path | None) -> Path:
    """Return the DM-FEM2D data root in CLI/env/repo-local priority order."""

    if data_root is not None:
        return Path(data_root)
    env_root = os.environ.get("RODM_DM_FEM_ROOT")
    if env_root:
        return Path(env_root)
    if REPO_LOCAL_DATA_ROOT.exists():
        return REPO_LOCAL_DATA_ROOT
    return Path.home() / "data" / "DM-FEM2D"


def build_default_case(
    data_root: str | Path | None,
    *,
    reversed_hydro: bool,
    structural_reduction_method: str = "serep_ridge",
    serep_ridge_relative_lambda: float = 1.0e-16,
) -> RodmFrequencyCase:
    """Build the documented 10-module 300 m reference case."""

    root = default_dm_fem_root(data_root)
    return RodmFrequencyCase(
        case_id="reference_case_300",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=root / "HydrodynamicData" / "Yoga" / "DM10_300_direction0.nc",
        structural_matrices=StructuralMatrixPaths(
            mass=root / "StructureData" / "JobMesh5_5_MASS1.mtx",
            stiffness=root / "StructureData" / "JobMesh5_5_STIF1.mtx",
        ),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        mass_blend_beta=0.0,
        structural_reduction_method=structural_reduction_method,
        serep_ridge_relative_lambda=serep_ridge_relative_lambda,
        use_hydrostatic=True,
        frequency_index=0,
        reverse_hydrodynamic_node_order=reversed_hydro,
    )


def missing_inputs(case: RodmFrequencyCase) -> list[Path]:
    """Return missing files for the basic RODM time-domain run."""

    candidates = (
        case.hydrodynamic_dataset,
        case.structural_matrices.mass,
        case.structural_matrices.stiffness,
    )
    return [path for path in candidates if not Path(path).exists()]


def centerline_heave_time(global_displacement: np.ndarray, *, retained_dofs_per_node: int) -> np.ndarray:
    """Extract centerline heave time history with shape ``(n_time, 60)``."""

    response = np.asarray(global_displacement)
    if response.ndim != 2:
        raise ValueError("global_displacement must have shape (n_time, ndof)")
    indices = [
        (node - 1) * retained_dofs_per_node + HEAVE_DOF_ZERO_BASED
        for node in range(CENTERLINE_START_NODE, CENTERLINE_STOP_NODE_EXCLUSIVE)
    ]
    return response[:, indices]


def representative_columns(count: int) -> tuple[int, int, int]:
    """Return bow/mid/stern-like column indices over the extracted centerline."""

    if count < 3:
        raise ValueError("centerline must contain at least three points")
    return (0, count // 2, count - 1)


def write_representative_csv(path: Path, time: np.ndarray, heave: np.ndarray) -> Path:
    """Write representative heave histories to CSV."""

    columns = representative_columns(heave.shape[1])
    labels = ("x0", "x_mid", "x1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", *labels))
        for row_index, t in enumerate(time):
            writer.writerow([float(t), *(float(heave[row_index, col]) for col in columns)])
    return path


def plot_representative_heave(output_dir: Path, time: np.ndarray, heave: np.ndarray) -> Path:
    """Plot representative centerline heave histories."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = representative_columns(heave.shape[1])
    labels = ("x/L = 0", "x/L = 0.5", "x/L = 1")
    colors = ("#1f77b4", "#d62728", "#2ca02c")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for column, label, color in zip(columns, labels, colors):
        ax.plot(time, heave[:, column], linewidth=1.2, color=color, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heave displacement")
    ax.set_title("RODM time-domain centerline heave histories")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "centerline_representative_heave_time.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_heave_snapshots(
    output_dir: Path,
    time: np.ndarray,
    heave: np.ndarray,
    *,
    sample_count: int = 6,
) -> Path:
    """Plot several centerline heave snapshots over one late-time window."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    if time.size < sample_count:
        indices = np.arange(time.size)
    else:
        start = int(0.75 * (time.size - 1))
        indices = np.linspace(start, time.size - 1, sample_count, dtype=int)
    x = np.linspace(0.0, 1.0, heave.shape[1])

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for index in indices:
        ax.plot(x, heave[index], linewidth=1.1, label=f"t={time[index]:.1f}s")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave displacement")
    ax.set_title("RODM time-domain centerline heave snapshots")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output_dir / "centerline_heave_snapshots.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_frequency_validation(
    output_dir: Path,
    frequency_amplitude: np.ndarray,
    fitted_time_amplitude: np.ndarray,
    *,
    retained_dofs_per_node: int,
) -> Path:
    """Plot fitted time-domain centerline amplitude against frequency-domain amplitude."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    freq_heave = centerline_heave_time(
        frequency_amplitude.reshape(1, -1),
        retained_dofs_per_node=retained_dofs_per_node,
    )[0]
    time_heave = centerline_heave_time(
        fitted_time_amplitude.reshape(1, -1),
        retained_dofs_per_node=retained_dofs_per_node,
    )[0]
    x = np.linspace(0.0, 1.0, freq_heave.size)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(x, np.abs(freq_heave), color="#1f77b4", linewidth=1.7, label="frequency-domain")
    ax.plot(x, np.abs(time_heave), color="#d62728", linestyle="--", linewidth=1.5, label="time-domain fit")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave amplitude")
    ax.set_title("RODM time-domain steady-state amplitude check")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "centerline_heave_frequency_validation.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_memory_force_norm(output_dir: Path, time: np.ndarray, memory_force: np.ndarray) -> Path:
    """Plot the Euclidean norm of the radiation-memory force."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    force_norm = np.linalg.norm(memory_force, axis=1)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(time, force_norm, color="#9467bd", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory-force norm")
    ax.set_title("RODM radiation-memory force history")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = output_dir / "radiation_memory_force_norm.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_radiation_irf_norm(output_dir: Path, irf_time: np.ndarray, radiation_irf: np.ndarray) -> Path:
    """Plot the Frobenius norm of the radiation impulse-response function."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    irf_norm = np.linalg.norm(radiation_irf.reshape(radiation_irf.shape[0], -1), axis=1)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(irf_time, irf_norm, color="#8c564b", linewidth=1.2)
    ax.set_xlabel("Memory time (s)")
    ax.set_ylabel("IRF Frobenius norm")
    ax.set_title("RODM radiation IRF decay")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    path = output_dir / "radiation_irf_norm.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def write_report(output_root: Path, metrics: dict[str, object]) -> Path:
    """Write a short Markdown report for the completed or skipped run."""

    path = output_root / "report.md"
    lines = [
        "# 300 m 基础 RODM 时域时序算例",
        "",
        f"生成时间：{metrics['generated_at']}",
        "",
        "## 状态",
        "",
        f"- 状态：`{metrics['status']}`",
        f"- 算例：`{metrics['case_id']}`",
        f"- 水动力节点反序：`{metrics.get('reverse_hydrodynamic_node_order')}`",
        "",
    ]
    if metrics["status"] == "missing_inputs":
        lines.extend(["## 缺失输入", ""])
        lines.extend(f"- `{path}`" for path in metrics["missing_inputs"])
        lines.extend(
            [
                "",
                "设置 `RODM_DM_FEM_ROOT` 或使用 `--data-root` 指向 DM-FEM2D 数据目录后重新运行。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 输出文件",
                "",
                f"- 时间：`{metrics['time_path']}`",
                f"- 全局位移时序：`{metrics['global_displacement_path']}`",
                f"- 主自由度位移时序：`{metrics['master_displacement_path']}`",
                f"- 中心线 heave 时序：`{metrics['centerline_heave_path']}`",
                f"- 代表点 CSV：`{metrics['representative_csv']}`",
                "",
                "## 时域参数",
                "",
                f"- `omega_rad_s`: `{metrics['omega_rad_s']}`",
                f"- `period_s`: `{metrics['period_s']}`",
                f"- `time_step_s`: `{metrics['time_step_s']}`",
                f"- `duration_s`: `{metrics['duration_s']}`",
                f"- `time_samples`: `{metrics['time_samples']}`",
                "",
                "## 图件",
                "",
            ]
        )
        lines.extend(f"- `{figure}`" for figure in metrics["figures"])
        if "global_amplitude_error" in metrics:
            lines.extend(
                [
                    "",
                    "## 频域稳态对比",
                    "",
                    f"- 全局 DOF L2 相对误差：`{metrics['global_amplitude_error']['l2_relative_error']}`",
                    f"- 主 DOF L2 相对误差：`{metrics['master_amplitude_error']['l2_relative_error']}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional RODM YAML config.")
    parser.add_argument("--data-root", default=None, help="DM-FEM2D data root for the default case.")
    parser.add_argument(
        "--hydro-node-order",
        choices=("default", "reversed"),
        default="reversed",
        help="Hydrodynamic node block order for the default case.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cycles", type=float, default=80.0)
    parser.add_argument("--steps-per-cycle", type=int, default=180)
    parser.add_argument("--ramp-cycles", type=float, default=5.0)
    parser.add_argument("--wave-amplitude", type=float, default=1.0)
    parser.add_argument("--phase-rad", type=float, default=0.0)
    parser.add_argument(
        "--structural-reduction-method",
        choices=("serep", "serep_ridge", "serep_robust", "guyan_static"),
        default="serep_ridge",
        help="Structural reduction method. serep_ridge is the stable time-domain default.",
    )
    parser.add_argument(
        "--serep-ridge-relative-lambda",
        type=float,
        default=1.0e-16,
        help="Relative Tikhonov regularization used by structural_reduction_method=serep_ridge.",
    )
    parser.add_argument(
        "--radiation-model",
        choices=("constant", "direct_convolution"),
        default="constant",
        help="Time-domain radiation model.",
    )
    parser.add_argument(
        "--memory-duration",
        type=float,
        default=None,
        help="Direct-convolution memory-kernel duration in seconds.",
    )
    parser.add_argument(
        "--damping-convention",
        choices=("physical", "wec_sim_bemio"),
        default="physical",
        help="Radiation damping convention used to build the IRF.",
    )
    parser.add_argument(
        "--infinite-added-mass-method",
        choices=("high_frequency", "ogilvie"),
        default="high_frequency",
        help="Method used to estimate infinite-frequency added mass.",
    )
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default="none",
        help="Correction applied to radiation damping before IRF generation.",
    )
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default="none",
        help="Optional finite-band residual correction for regular-wave validation.",
    )
    parser.add_argument(
        "--radiation-frequency-window",
        choices=("none", "linear_tail", "cosine_tail"),
        default="none",
        help="Optional high-frequency damping taper before IRF generation.",
    )
    parser.add_argument("--radiation-window-start-omega", type=float, default=None)
    parser.add_argument("--radiation-window-stop-omega", type=float, default=None)
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default="rectangular",
        help="Discrete quadrature rule used for the radiation-memory convolution.",
    )
    parser.add_argument(
        "--added-mass-tail-count",
        type=int,
        default=3,
        help="Number of high-frequency added-mass matrices to average for A_inf.",
    )
    parser.add_argument("--discard-cycles", type=float, default=55.0)
    parser.add_argument(
        "--skip-frequency-validation",
        action="store_true",
        help="Only compute time series; skip fitted-amplitude comparison.",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        case = build_rodm_frequency_case_from_config(args.config)
        if args.structural_reduction_method is not None:
            case = replace(
                case,
                structural_reduction_method=args.structural_reduction_method,
                serep_ridge_relative_lambda=args.serep_ridge_relative_lambda,
            )
    else:
        case = build_default_case(
            args.data_root,
            reversed_hydro=args.hydro_node_order == "reversed",
            structural_reduction_method=args.structural_reduction_method,
            serep_ridge_relative_lambda=args.serep_ridge_relative_lambda,
        )

    missing = missing_inputs(case)
    if missing:
        metrics = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "missing_inputs",
            "case_id": case.case_id,
            "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
            "structural_reduction_method": case.structural_reduction_method,
            "serep_ridge_relative_lambda": case.serep_ridge_relative_lambda,
            "missing_inputs": missing,
        }
        metrics_path = write_metrics_json(output_root / "metrics.json", metrics)
        report_path = write_report(output_root, metrics)
        print("Basic time-domain case skipped because inputs are missing.")
        for path in missing:
            print(f"missing: {path}")
        print(f"metrics: {metrics_path}")
        print(f"report: {report_path}")
        return 1 if args.fail_on_missing else 0

    import xarray as xr

    dataset = xr.open_dataset(case.hydrodynamic_dataset)
    try:
        omega = float(np.ravel(dataset.omega.values)[case.frequency_index])
    finally:
        dataset.close()

    period = 2.0 * np.pi / omega
    time_config = TimeDomainSimulationConfig(
        time_step=period / args.steps_per_cycle,
        duration=args.cycles * period,
        wave_amplitude=args.wave_amplitude,
        phase_rad=args.phase_rad,
        ramp_time=args.ramp_cycles * period,
        radiation_model=args.radiation_model,
        memory_duration=args.memory_duration,
        damping_convention=args.damping_convention,
        infinite_added_mass_method=args.infinite_added_mass_method,
        added_mass_tail_count=args.added_mass_tail_count,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_residual_model=args.radiation_residual_model,
        radiation_frequency_window=args.radiation_frequency_window,
        radiation_window_start_omega=args.radiation_window_start_omega,
        radiation_window_stop_omega=args.radiation_window_stop_omega,
        radiation_convolution_rule=args.radiation_convolution_rule,
    )

    start = timer.perf_counter()
    result = solve_rodm_time_domain_case(case, time_config)
    elapsed = timer.perf_counter() - start
    heave = centerline_heave_time(
        result.global_displacement,
        retained_dofs_per_node=case.retained_dofs_per_node,
    )

    time_path = output_root / "time.npy"
    global_path = output_root / "global_displacement_time.npy"
    master_path = output_root / "master_displacement_time.npy"
    heave_path = output_root / "centerline_heave_time.npy"
    velocity_path = output_root / "master_velocity_time.npy"
    acceleration_path = output_root / "master_acceleration_time.npy"
    memory_force_path = output_root / "memory_force_time.npy"
    np.save(time_path, result.time)
    np.save(global_path, result.global_displacement)
    np.save(master_path, result.master_displacement)
    np.save(heave_path, heave)
    np.save(velocity_path, result.master_velocity)
    np.save(acceleration_path, result.master_acceleration)
    np.save(memory_force_path, result.memory_force)
    radiation_irf_time_path = None
    radiation_irf_path = None
    added_mass_infinite_path = None
    if result.radiation_irf_time is not None:
        radiation_irf_time_path = output_root / "radiation_irf_time.npy"
        np.save(radiation_irf_time_path, result.radiation_irf_time)
    if result.radiation_irf is not None:
        radiation_irf_path = output_root / "radiation_irf.npy"
        np.save(radiation_irf_path, result.radiation_irf)
    if result.added_mass_infinite is not None:
        added_mass_infinite_path = output_root / "added_mass_infinite.npy"
        np.save(added_mass_infinite_path, result.added_mass_infinite)

    representative_csv = write_representative_csv(
        output_root / "centerline_representative_heave.csv",
        result.time,
        heave,
    )
    figures = [
        plot_representative_heave(output_root / "figures", result.time, heave),
        plot_heave_snapshots(output_root / "figures", result.time, heave),
    ]
    if np.any(result.memory_force):
        figures.append(
            plot_memory_force_norm(
                output_root / "figures",
                result.time,
                result.memory_force,
            )
        )
    if result.radiation_irf_time is not None and result.radiation_irf is not None:
        figures.append(
            plot_radiation_irf_norm(
                output_root / "figures",
                result.radiation_irf_time,
                result.radiation_irf,
            )
        )

    metrics: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "case_id": case.case_id,
        "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
        "structural_reduction_method": case.structural_reduction_method,
        "serep_ridge_relative_lambda": case.serep_ridge_relative_lambda,
        "omega_rad_s": omega,
        "period_s": period,
        "time_step_s": time_config.time_step,
        "duration_s": time_config.duration,
        "radiation_model": time_config.radiation_model,
        "memory_duration_s": time_config.memory_duration,
        "damping_convention": time_config.damping_convention,
        "infinite_added_mass_method": time_config.infinite_added_mass_method,
        "added_mass_tail_count": time_config.added_mass_tail_count,
        "radiation_passivity_correction": time_config.radiation_passivity_correction,
        "cycles": args.cycles,
        "steps_per_cycle": args.steps_per_cycle,
        "ramp_cycles": args.ramp_cycles,
        "wave_amplitude": args.wave_amplitude,
        "phase_rad": args.phase_rad,
        "time_samples": int(result.time.size),
        "global_displacement_shape": result.global_displacement.shape,
        "centerline_heave_shape": heave.shape,
        "elapsed_seconds": elapsed,
        "time_path": time_path,
        "global_displacement_path": global_path,
        "master_displacement_path": master_path,
        "master_velocity_path": velocity_path,
        "master_acceleration_path": acceleration_path,
        "memory_force_path": memory_force_path,
        "radiation_irf_time_path": radiation_irf_time_path,
        "radiation_irf_path": radiation_irf_path,
        "added_mass_infinite_path": added_mass_infinite_path,
        "centerline_heave_path": heave_path,
        "representative_csv": representative_csv,
        "figures": figures,
    }

    if not args.skip_frequency_validation:
        start = timer.perf_counter()
        frequency = solve_rodm_frequency_case(case)
        metrics["frequency_elapsed_seconds"] = timer.perf_counter() - start
        fitted_global = fit_harmonic_amplitude(
            result.global_displacement,
            result.time,
            omega,
            start_time=args.discard_cycles * period,
        )
        fitted_master = fit_harmonic_amplitude(
            result.master_displacement,
            result.time,
            omega,
            start_time=args.discard_cycles * period,
        )
        frequency_global = args.wave_amplitude * frequency.global_displacement.reshape(-1)
        frequency_master = args.wave_amplitude * frequency.master_displacement.reshape(-1)
        fitted_global_path = output_root / "fitted_global_amplitude.npy"
        frequency_global_path = output_root / "frequency_global_amplitude.npy"
        np.save(fitted_global_path, fitted_global)
        np.save(frequency_global_path, frequency_global)
        metrics["fitted_global_amplitude_path"] = fitted_global_path
        metrics["frequency_global_amplitude_path"] = frequency_global_path
        metrics["global_amplitude_error"] = harmonic_amplitude_error(
            fitted_global,
            frequency_global,
        )
        metrics["master_amplitude_error"] = harmonic_amplitude_error(
            fitted_master,
            frequency_master,
        )
        figures.append(
            plot_frequency_validation(
                output_root / "figures",
                frequency_global,
                fitted_global,
                retained_dofs_per_node=case.retained_dofs_per_node,
            )
        )

    metrics_path = write_metrics_json(output_root / "metrics.json", metrics)
    report_path = write_report(output_root, metrics)

    print("Basic RODM time-domain time-series run completed.")
    print(f"time_samples: {result.time.size}")
    print(f"global_displacement_shape: {result.global_displacement.shape}")
    print(f"centerline_heave_shape: {heave.shape}")
    if "global_amplitude_error" in metrics:
        print(
            "global_l2_relative_error: "
            f"{metrics['global_amplitude_error']['l2_relative_error']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
