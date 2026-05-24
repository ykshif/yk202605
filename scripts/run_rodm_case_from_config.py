"""Run a RODM case from a YAML configuration file.

The default domain is frequency-domain to preserve the original workflow. Use
``--domain time`` to run the first linear time-domain RODM path with the same
case configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import build_rodm_frequency_case_from_config  # noqa: E402
from offshore_energy_sim.core import load_case_config  # noqa: E402
from offshore_energy_sim.core import build_workflow_paths, write_metrics_json  # noqa: E402
from offshore_energy_sim.mooring import (  # noqa: E402
    build_mooring_provider_from_config,
    is_mooring_enabled,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    solve_rodm_time_domain_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RODM case from YAML configuration in frequency or time domain.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "reference_case_300.yaml",
        help="Path to the case YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Frequency: response. Time: global displacement history.",
    )
    parser.add_argument(
        "--case-output-dir",
        type=Path,
        default=None,
        help="Case output directory. Defaults to results/<case_id>.",
    )
    parser.add_argument(
        "--domain",
        choices=("frequency", "time"),
        default=None,
        help="Simulation domain. Defaults to solver.domain/type in config, then frequency.",
    )
    parser.add_argument(
        "--reverse-hydrodynamic-node-order",
        action="store_true",
        default=None,
        help="Reverse hydrodynamic node blocks before solving.",
    )
    parser.add_argument("--time-step", type=float, default=None, help="Time-domain step size in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Time-domain duration in seconds.")
    parser.add_argument(
        "--cycles",
        type=float,
        default=None,
        help="Time-domain duration in wave cycles when --duration is omitted.",
    )
    parser.add_argument(
        "--steps-per-cycle",
        type=int,
        default=None,
        help="Time-domain time-step resolution when --time-step is omitted.",
    )
    parser.add_argument(
        "--ramp-cycles",
        type=float,
        default=None,
        help="Cosine-ramp duration in wave cycles for time-domain regular-wave forcing.",
    )
    parser.add_argument(
        "--wave-amplitude",
        type=float,
        default=None,
        help="Regular-wave amplitude multiplier for time-domain forcing.",
    )
    parser.add_argument(
        "--phase-rad",
        type=float,
        default=None,
        help="Regular-wave phase in radians for time-domain forcing.",
    )
    parser.add_argument(
        "--radiation-model",
        choices=("constant", "direct_convolution"),
        default=None,
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
        default=None,
        help="Radiation damping convention used to build the IRF.",
    )
    parser.add_argument(
        "--infinite-added-mass-method",
        choices=("high_frequency", "ogilvie"),
        default=None,
        help="Method used to estimate infinite-frequency added mass.",
    )
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default=None,
        help="Correction applied to radiation damping before IRF generation.",
    )
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default=None,
        help="Optional finite-band residual correction for regular-wave validation.",
    )
    parser.add_argument(
        "--radiation-frequency-window",
        choices=("none", "linear_tail", "cosine_tail"),
        default=None,
        help="Optional high-frequency damping taper before IRF generation.",
    )
    parser.add_argument(
        "--radiation-window-start-omega",
        type=float,
        default=None,
        help="Start angular frequency for the optional radiation taper.",
    )
    parser.add_argument(
        "--radiation-window-stop-omega",
        type=float,
        default=None,
        help="Stop angular frequency for the optional radiation taper.",
    )
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default=None,
        help="Discrete quadrature rule used for the radiation-memory convolution.",
    )
    parser.add_argument(
        "--added-mass-tail-count",
        type=int,
        default=None,
        help="Number of high-frequency added-mass matrices to average for A_inf.",
    )
    return parser.parse_args()


def selected_domain(config_path: Path, override: str | None) -> str:
    """Return frequency/time selection from CLI or config."""

    if override is not None:
        return override
    config = load_case_config(config_path)
    solver = config.get("solver", {})
    domain = str(solver.get("domain", "")).lower()
    solver_type = str(solver.get("type", "")).lower()
    if domain in {"frequency", "time"}:
        return domain
    if solver_type.startswith("time"):
        return "time"
    return "frequency"


def selected_case_omega(case) -> float:
    """Read the selected angular frequency without loading all BEM variables."""

    import xarray as xr

    dataset = xr.open_dataset(case.hydrodynamic_dataset)
    try:
        return float(np.ravel(dataset.omega.values)[case.frequency_index])
    finally:
        dataset.close()


def time_settings_from_config(config_path: Path) -> dict[str, object]:
    """Return the optional time_domain config section."""

    config = load_case_config(config_path)
    settings = config.get("time_domain", {})
    if not isinstance(settings, dict):
        raise ValueError("time_domain config section must be a mapping")
    return settings


def _setting(args: argparse.Namespace, settings: dict[str, object], name: str, default):
    """Return CLI value, config value, or default in that priority order."""

    value = getattr(args, name)
    if value is not None:
        return value
    return settings.get(name, default)


def time_config_from_args(
    args: argparse.Namespace,
    omega: float,
    settings: dict[str, object],
) -> TimeDomainSimulationConfig:
    """Build a time-domain configuration from CLI controls."""

    if omega <= 0.0:
        raise ValueError("time-domain run requires a positive selected omega")
    period = 2.0 * np.pi / omega
    cycles = float(_setting(args, settings, "cycles", 80.0))
    steps_per_cycle = int(_setting(args, settings, "steps_per_cycle", 180))
    ramp_cycles = float(_setting(args, settings, "ramp_cycles", 5.0))
    wave_amplitude = float(_setting(args, settings, "wave_amplitude", 1.0))
    phase_rad = float(_setting(args, settings, "phase_rad", 0.0))
    radiation_model = str(_setting(args, settings, "radiation_model", "constant"))
    memory_duration_value = _setting(args, settings, "memory_duration", None)
    memory_duration = (
        None
        if memory_duration_value is None
        else float(memory_duration_value)
    )
    damping_convention = str(_setting(args, settings, "damping_convention", "physical"))
    infinite_added_mass_method = str(
        _setting(args, settings, "infinite_added_mass_method", "high_frequency")
    )
    radiation_passivity_correction = str(
        _setting(args, settings, "radiation_passivity_correction", "none")
    )
    radiation_residual_model = str(
        _setting(args, settings, "radiation_residual_model", "none")
    )
    radiation_frequency_window = str(
        _setting(args, settings, "radiation_frequency_window", "none")
    )
    radiation_window_start_omega_value = _setting(
        args,
        settings,
        "radiation_window_start_omega",
        None,
    )
    radiation_window_start_omega = (
        None
        if radiation_window_start_omega_value is None
        else float(radiation_window_start_omega_value)
    )
    radiation_window_stop_omega_value = _setting(
        args,
        settings,
        "radiation_window_stop_omega",
        None,
    )
    radiation_window_stop_omega = (
        None
        if radiation_window_stop_omega_value is None
        else float(radiation_window_stop_omega_value)
    )
    radiation_convolution_rule = str(
        _setting(args, settings, "radiation_convolution_rule", "rectangular")
    )
    added_mass_tail_count = int(_setting(args, settings, "added_mass_tail_count", 3))
    if steps_per_cycle < 2:
        raise ValueError("--steps-per-cycle must be at least 2")
    time_step = args.time_step if args.time_step is not None else period / steps_per_cycle
    duration = args.duration if args.duration is not None else cycles * period
    return TimeDomainSimulationConfig(
        time_step=time_step,
        duration=duration,
        wave_amplitude=wave_amplitude,
        phase_rad=phase_rad,
        ramp_time=ramp_cycles * period,
        radiation_model=radiation_model,
        memory_duration=memory_duration,
        damping_convention=damping_convention,
        infinite_added_mass_method=infinite_added_mass_method,
        added_mass_tail_count=added_mass_tail_count,
        radiation_passivity_correction=radiation_passivity_correction,
        radiation_residual_model=radiation_residual_model,
        radiation_frequency_window=radiation_frequency_window,
        radiation_window_start_omega=radiation_window_start_omega,
        radiation_window_stop_omega=radiation_window_stop_omega,
        radiation_convolution_rule=radiation_convolution_rule,
    )


def selected_radiation_model(args: argparse.Namespace, settings: dict[str, object]) -> str:
    """Return the time-domain radiation model for output variant naming."""

    return str(_setting(args, settings, "radiation_model", "constant")).lower()


def variant_id_for(
    domain: str,
    reverse_hydrodynamic_node_order: bool,
    *,
    radiation_model: str = "constant",
    mooring_enabled: bool = False,
) -> str:
    """Return a standard output variant name."""

    if domain == "frequency":
        return "hydro_reversed" if reverse_hydrodynamic_node_order else "default"
    base = "time_domain_hydro_reversed" if reverse_hydrodynamic_node_order else "time_domain"
    if radiation_model != "constant":
        base = f"{base}_{radiation_model}"
    if mooring_enabled:
        base = f"{base}_mooring"
    return base


def run_frequency_domain(case, paths, output: Path) -> dict[str, object]:
    """Run and save the existing frequency-domain workflow."""

    start = time.perf_counter()
    result = solve_rodm_frequency_case(case)
    elapsed = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result.global_displacement)
    return {
        "domain": "frequency",
        "response_shape": result.global_displacement.shape,
        "elapsed_seconds": elapsed,
        "response_path": output,
    }


def run_time_domain(
    case,
    args: argparse.Namespace,
    paths,
    output: Path,
    settings: dict[str, object],
    case_config: dict[str, object],
) -> dict[str, object]:
    """Run and save the time-domain workflow."""

    omega = selected_case_omega(case)
    config = time_config_from_args(args, omega, settings)
    mooring_provider = build_mooring_provider_from_config(case_config)
    start = time.perf_counter()
    result = solve_rodm_time_domain_case(
        case,
        config,
        mooring_provider=mooring_provider,
    )
    elapsed = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result.global_displacement)
    np.save(paths.variant_root / "time.npy", result.time)
    np.save(paths.variant_root / "master_displacement_time.npy", result.master_displacement)
    np.save(paths.variant_root / "master_velocity_time.npy", result.master_velocity)
    np.save(paths.variant_root / "master_acceleration_time.npy", result.master_acceleration)
    np.save(paths.variant_root / "memory_force_time.npy", result.memory_force)
    if result.radiation_irf_time is not None:
        np.save(paths.variant_root / "radiation_irf_time.npy", result.radiation_irf_time)
    if result.radiation_irf is not None:
        np.save(paths.variant_root / "radiation_irf.npy", result.radiation_irf)
    if result.added_mass_infinite is not None:
        np.save(paths.variant_root / "added_mass_infinite.npy", result.added_mass_infinite)
    mooring_metrics = save_mooring_outputs(paths.variant_root, result)
    return {
        "domain": "time",
        "radiation_model": config.radiation_model,
        "omega_rad_s": omega,
        "time_step": config.time_step,
        "duration": config.duration,
        "memory_duration": config.memory_duration,
        "damping_convention": config.damping_convention,
        "infinite_added_mass_method": config.infinite_added_mass_method,
        "added_mass_tail_count": config.added_mass_tail_count,
        "radiation_passivity_correction": config.radiation_passivity_correction,
        "time_samples": int(result.time.size),
        "wave_amplitude": config.wave_amplitude,
        "phase_rad": config.phase_rad,
        "ramp_time": config.ramp_time,
        "response_shape": result.global_displacement.shape,
        "elapsed_seconds": elapsed,
        "response_path": output,
        "time_path": paths.variant_root / "time.npy",
        "master_displacement_path": paths.variant_root / "master_displacement_time.npy",
        "mooring": mooring_metrics,
    }


def save_mooring_outputs(variant_root: Path, result) -> dict[str, object]:
    """Save optional reduced mooring terms and return JSON-friendly metrics."""

    metadata = result.mooring_metadata or {"enabled": False}
    summary: dict[str, object] = {
        "enabled": bool(metadata.get("enabled", False)),
        "metadata": metadata,
    }
    if result.mooring_reduced_stiffness is not None:
        path = variant_root / "mooring_reduced_stiffness.npy"
        np.save(path, result.mooring_reduced_stiffness)
        summary["reduced_stiffness_path"] = path
        summary["reduced_stiffness_frobenius_norm"] = float(
            np.linalg.norm(result.mooring_reduced_stiffness)
        )
        summary["reduced_stiffness_trace"] = float(np.trace(result.mooring_reduced_stiffness))
    if result.mooring_reduced_damping is not None:
        path = variant_root / "mooring_reduced_damping.npy"
        np.save(path, result.mooring_reduced_damping)
        summary["reduced_damping_path"] = path
        summary["reduced_damping_frobenius_norm"] = float(
            np.linalg.norm(result.mooring_reduced_damping)
        )
        summary["reduced_damping_trace"] = float(np.trace(result.mooring_reduced_damping))
    if result.mooring_reduced_pretension is not None:
        path = variant_root / "mooring_reduced_pretension.npy"
        np.save(path, result.mooring_reduced_pretension)
        summary["reduced_pretension_path"] = path
        summary["reduced_pretension_norm"] = float(
            np.linalg.norm(result.mooring_reduced_pretension)
        )
    return summary


def main() -> int:
    args = parse_args()
    domain = selected_domain(args.config, args.domain)
    case_config = load_case_config(args.config)
    case = build_rodm_frequency_case_from_config(
        args.config,
        reverse_hydrodynamic_node_order=args.reverse_hydrodynamic_node_order,
    )
    time_settings = time_settings_from_config(args.config) if domain == "time" else {}
    radiation_model = (
        selected_radiation_model(args, time_settings)
        if domain == "time"
        else "constant"
    )

    variant_id = variant_id_for(
        domain,
        case.reverse_hydrodynamic_node_order,
        radiation_model=radiation_model,
        mooring_enabled=(domain == "time" and is_mooring_enabled(case_config)),
    )
    case_output_dir = args.case_output_dir or (REPO_ROOT / "results" / case.case_id)
    paths = build_workflow_paths(case_output_dir, variant_id=variant_id)
    output = Path(args.output) if args.output is not None else paths.response_path

    if domain == "frequency":
        metrics = run_frequency_domain(case, paths, output)
        if is_mooring_enabled(case_config):
            metrics["mooring"] = {
                "enabled": True,
                "applied": False,
                "reason": "frequency-domain mooring is not enabled in this runner yet",
            }
    else:
        metrics = run_time_domain(case, args, paths, output, time_settings, case_config)

    write_metrics_json(
        paths.metrics_path,
        {
            "case_id": case.case_id,
            "variant_id": variant_id,
            "reverse_hydrodynamic_node_order": case.reverse_hydrodynamic_node_order,
            **metrics,
        },
    )

    print(f"case_id: {case.case_id}")
    print(f"domain: {domain}")
    print(f"variant_id: {variant_id}")
    print(f"reverse_hydrodynamic_node_order: {case.reverse_hydrodynamic_node_order}")
    print(f"response_shape: {metrics['response_shape']}")
    print(f"elapsed_seconds: {metrics['elapsed_seconds']:.3f}")
    print(f"wrote: {output}")
    print(f"metrics: {paths.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
