"""Validate the RODM time-domain solver against the frequency-domain response.

The default case is the 300 m x 60 m, 10 hydrodynamic-node reference model.
When the external DM-FEM2D data tree is not available, the script reports the
missing inputs and exits successfully unless ``--fail-on-missing`` is used.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
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
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    fit_harmonic_amplitude,
    harmonic_amplitude_error,
    solve_rodm_time_domain_case,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "rodm_single_frequency"
REPO_LOCAL_DATA_ROOT = REPO_ROOT / "data" / "external" / "DM-FEM2D"


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


def build_default_300m_case(
    data_root: str | Path | None,
    *,
    reversed_hydro: bool,
    structural_reduction_method: str = "serep_ridge",
    serep_ridge_relative_lambda: float = 1.0e-16,
) -> RodmFrequencyCase:
    """Build the standard 10-module 300 m RODM case from a data root."""

    root = default_dm_fem_root(data_root)
    return RodmFrequencyCase(
        case_id="time_domain_reference_300m",
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
    """Return missing inputs required by the RODM solve."""

    paths = (
        case.hydrodynamic_dataset,
        case.structural_matrices.mass,
        case.structural_matrices.stiffness,
    )
    return [path for path in paths if not Path(path).exists()]


def plot_heave_comparison(
    frequency_response: np.ndarray,
    fitted_time_response: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot frequency-domain heave amplitude against fitted time-domain amplitude."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x_frequency, y_frequency = extract_centerline_heave(frequency_response.reshape(-1, 1))
    x_time, y_time = extract_centerline_heave(fitted_time_response.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x_frequency, y_frequency, color="#1f77b4", linewidth=1.7, label="frequency-domain")
    ax.plot(x_time, y_time, color="#d62728", linestyle="--", linewidth=1.5, label="time-domain fit")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title("RODM time-domain steady-state validation")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "heave_frequency_vs_time_fit.png"
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional RODM YAML config.")
    parser.add_argument("--data-root", default=None, help="DM-FEM2D data root for the default 300 m case.")
    parser.add_argument(
        "--hydro-node-order",
        choices=("default", "reversed"),
        default="reversed",
        help="Hydrodynamic node block order for the default 300 m case.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cycles", type=float, default=80.0)
    parser.add_argument("--discard-cycles", type=float, default=55.0)
    parser.add_argument("--steps-per-cycle", type=int, default=180)
    parser.add_argument("--ramp-cycles", type=float, default=5.0)
    parser.add_argument("--wave-amplitude", type=float, default=1.0)
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
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config is not None:
        case = build_rodm_frequency_case_from_config(args.config)
        if args.structural_reduction_method is not None:
            case = replace(
                case,
                structural_reduction_method=args.structural_reduction_method,
                serep_ridge_relative_lambda=args.serep_ridge_relative_lambda,
            )
    else:
        case = build_default_300m_case(
            args.data_root,
            reversed_hydro=args.hydro_node_order == "reversed",
            structural_reduction_method=args.structural_reduction_method,
            serep_ridge_relative_lambda=args.serep_ridge_relative_lambda,
        )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    missing = missing_inputs(case)
    if missing:
        metrics = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "missing_inputs",
            "case_id": case.case_id,
            "structural_reduction_method": case.structural_reduction_method,
            "serep_ridge_relative_lambda": case.serep_ridge_relative_lambda,
            "missing_inputs": missing,
            "note": "Set RODM_DM_FEM_ROOT or pass --data-root to run the full validation.",
        }
        metrics_path = output_root / "metrics.json"
        write_metrics_json(metrics_path, metrics)
        print("Time-domain RODM validation skipped because inputs are missing.")
        for path in missing:
            print(f"missing: {path}")
        print(f"metrics: {metrics_path}")
        return 1 if args.fail_on_missing else 0

    start = timer.perf_counter()
    frequency = solve_rodm_frequency_case(case)
    frequency_elapsed = timer.perf_counter() - start
    omega = float(np.asarray(frequency.omega).reshape(-1)[0])
    period = 2.0 * np.pi / omega
    time_step = period / args.steps_per_cycle
    config = TimeDomainSimulationConfig(
        time_step=time_step,
        duration=args.cycles * period,
        wave_amplitude=args.wave_amplitude,
        ramp_time=args.ramp_cycles * period,
    )

    start = timer.perf_counter()
    time_domain = solve_rodm_time_domain_case(case, config)
    time_elapsed = timer.perf_counter() - start
    fitted_global = fit_harmonic_amplitude(
        time_domain.global_displacement,
        time_domain.time,
        omega,
        start_time=args.discard_cycles * period,
    )
    reference_global = args.wave_amplitude * frequency.global_displacement.reshape(-1)
    global_error = harmonic_amplitude_error(fitted_global, reference_global)

    fitted_master = fit_harmonic_amplitude(
        time_domain.master_displacement,
        time_domain.time,
        omega,
        start_time=args.discard_cycles * period,
    )
    reference_master = args.wave_amplitude * frequency.master_displacement.reshape(-1)
    master_error = harmonic_amplitude_error(fitted_master, reference_master)

    figure_path = plot_heave_comparison(
        reference_global,
        fitted_global,
        output_root / "figures",
    )
    np.save(output_root / "time.npy", time_domain.time)
    np.save(output_root / "global_displacement_time.npy", time_domain.global_displacement)
    np.save(output_root / "master_displacement_time.npy", time_domain.master_displacement)
    np.save(output_root / "fitted_global_amplitude.npy", fitted_global)
    np.save(output_root / "frequency_global_amplitude.npy", reference_global)

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "validated",
        "case_id": case.case_id,
        "omega_rad_s": omega,
        "period_s": period,
        "time_step_s": time_step,
        "cycles": args.cycles,
        "discard_cycles": args.discard_cycles,
        "steps_per_cycle": args.steps_per_cycle,
        "wave_amplitude": args.wave_amplitude,
        "structural_reduction_method": case.structural_reduction_method,
        "serep_ridge_relative_lambda": case.serep_ridge_relative_lambda,
        "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
        "frequency_elapsed_seconds": frequency_elapsed,
        "time_domain_elapsed_seconds": time_elapsed,
        "global_amplitude_error": global_error,
        "master_amplitude_error": master_error,
        "figure": figure_path,
    }
    metrics_path = output_root / "metrics.json"
    write_metrics_json(metrics_path, metrics)

    print("Time-domain RODM validation completed.")
    print(f"omega_rad_s: {omega:.8g}")
    print(f"global_l2_relative_error: {global_error['l2_relative_error']:.6g}")
    print(f"master_l2_relative_error: {master_error['l2_relative_error']:.6g}")
    print(f"figure: {figure_path}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
