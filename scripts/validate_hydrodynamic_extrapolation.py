"""Validate adapter-layer hydrodynamic extrapolation and radiation kernels."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import write_metrics_json  # noqa: E402
from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    HydrodynamicExtrapolationConfig,
    build_radiation_kernel,
    compare_radiation_kernels,
    extrapolate_hydrodynamic_data,
    frequency_grid_diagnostics,
    hydrodynamic_array_diagnostics,
)
from offshore_energy_sim.time_domain_adapter.extrapolation_diagnostics import (  # noqa: E402
    load_merged_hydrodynamic_arrays,
    plot_excitation_force_extrapolation_comparison,
    plot_hydrodynamic_ab_comparison,
    plot_radiation_kernel_norm,
    write_extrapolated_hydrodynamic_dataset,
)
from offshore_energy_sim.time_domain_adapter.wecsim_like_validation import (  # noqa: E402
    compare_case_statistics,
)

from run_time_domain_reference_case_300 import default_dm_fem_root  # noqa: E402


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "hydrodynamic_extrapolation_dm10_mesh2"
BENCHMARK_300M_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=BENCHMARK_300M_OMEGA)
    parser.add_argument("--low-frequency-min", type=float, default=0.02)
    parser.add_argument("--high-frequency-max", type=float, default=8.0)
    parser.add_argument("--low-frequency-count", type=int, default=4)
    parser.add_argument("--high-frequency-count", type=int, default=96)
    parser.add_argument("--memory-cycles", type=float, default=4.0)
    parser.add_argument("--steps-per-peak-cycle", type=int, default=40)
    parser.add_argument("--peak-cycles", type=float, default=80.0)
    parser.add_argument("--discard-peak-cycles", type=float, default=5.0)
    parser.add_argument("--significant-wave-height", type=float, default=1.0)
    parser.add_argument("--spectrum-seed", type=int, default=20260522)
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default="clip_negative_eigenvalues",
    )
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default="trapezoidal",
    )
    parser.add_argument(
        "--radiation-residual-model",
        choices=("none", "selected_frequency"),
        default="none",
        help="Optional residual correction used by the adapter-layer Cummins solve.",
    )
    parser.add_argument(
        "--run-time-domain-comparison",
        action="store_true",
        help="Run original and extrapolated JONSWAP time-domain validation cases.",
    )
    return parser.parse_args()


def kernel_time_values(selected_omega: float, memory_cycles: float, steps_per_peak_cycle: int) -> np.ndarray:
    period = 2.0 * np.pi / selected_omega
    dt = period / float(steps_per_peak_cycle)
    duration = memory_cycles * period
    step_count = int(np.floor(duration / dt))
    return np.arange(step_count + 1, dtype=float) * dt


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_time_domain_pair(
    *,
    data_root: Path,
    original_hydro_file: Path,
    extended_hydro_file: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    original_case = output_root / "time_domain_original"
    extended_case = output_root / "time_domain_extrapolated"
    runner = REPO_ROOT / "scripts" / "run_time_domain_excitation_case.py"
    validator = REPO_ROOT / "scripts" / "validate_spectrum_time_domain_statistics.py"

    shared = [
        "--data-root",
        str(data_root),
        "--excitation-model",
        "wave_spectrum",
        "--significant-wave-height",
        str(args.significant_wave_height),
        "--spectrum-type",
        "jonswap",
        "--spectrum-seed",
        str(args.spectrum_seed),
        "--target-omega",
        str(args.target_omega),
        "--peak-cycles",
        str(args.peak_cycles),
        "--steps-per-peak-cycle",
        str(args.steps_per_peak_cycle),
        "--memory-cycles",
        str(args.memory_cycles),
        "--radiation-passivity-correction",
        args.radiation_passivity_correction,
        "--radiation-convolution-rule",
        args.radiation_convolution_rule,
        "--radiation-residual-model",
        args.radiation_residual_model,
    ]
    run_command(
        [
            sys.executable,
            str(runner),
            "--hydro-file",
            str(original_hydro_file),
            "--output-root",
            str(original_case),
            *shared,
        ]
    )
    run_command(
        [
            sys.executable,
            str(validator),
            "--case-root",
            str(original_case),
            "--discard-peak-cycles",
            str(args.discard_peak_cycles),
        ]
    )
    run_command(
        [
            sys.executable,
            str(runner),
            "--hydro-file",
            str(extended_hydro_file),
            "--output-root",
            str(extended_case),
            *shared,
        ]
    )
    run_command(
        [
            sys.executable,
            str(validator),
            "--case-root",
            str(extended_case),
            "--discard-peak-cycles",
            str(args.discard_peak_cycles),
        ]
    )
    comparison = compare_case_statistics(original_case, extended_case)
    comparison["original_case_root"] = original_case
    comparison["extrapolated_case_root"] = extended_case
    return comparison


def main() -> int:
    args = parse_args()
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file if args.hydro_file.is_absolute() else data_root / args.hydro_file
    output_root = args.output_root
    figures_dir = output_root / "figures"
    hydro_output = (output_root / "hydrodynamics" / f"{hydro_path.stem}_adapter_extrapolated.nc").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = HydrodynamicExtrapolationConfig(
        low_frequency_min=args.low_frequency_min,
        high_frequency_max=args.high_frequency_max,
        low_frequency_count=args.low_frequency_count,
        high_frequency_count=args.high_frequency_count,
    )
    extended_hydro_path, file_invariance = write_extrapolated_hydrodynamic_dataset(
        hydro_path,
        hydro_output,
        config,
    )

    original = load_merged_hydrodynamic_arrays(hydro_path)
    extended = load_merged_hydrodynamic_arrays(extended_hydro_path)
    extrapolated = extrapolate_hydrodynamic_data(
        original["omega"],
        original["added_mass"],
        original["radiation_damping"],
        wave_force=original["wave_force"],
        config=config,
    )
    array_invariance = extrapolated.invariance_report(
        original["added_mass"],
        original["radiation_damping"],
        original["wave_force"],
    )

    selected_omega = float(original["omega"][np.argmin(np.abs(original["omega"] - args.target_omega))])
    kernel_time = kernel_time_values(selected_omega, args.memory_cycles, args.steps_per_peak_cycle)
    kernel_before = build_radiation_kernel(
        original["omega"],
        original["radiation_damping"],
        kernel_time,
        passivity_correction=args.radiation_passivity_correction,
    )
    kernel_after = build_radiation_kernel(
        extended["omega"],
        extended["radiation_damping"],
        kernel_time,
        passivity_correction=args.radiation_passivity_correction,
    )
    kernel_comparison = compare_radiation_kernels(kernel_time, kernel_before, kernel_after)

    figures = [
        plot_radiation_kernel_norm(
            figures_dir / "radiation_kernel_before_extrapolation.png",
            kernel_time,
            kernel_before,
            title="Radiation kernel before extrapolation",
        ),
        plot_radiation_kernel_norm(
            figures_dir / "radiation_kernel_after_extrapolation.png",
            kernel_time,
            kernel_after,
            title="Radiation kernel after extrapolation",
        ),
        plot_hydrodynamic_ab_comparison(
            figures_dir / "hydrodynamic_A_B_comparison.png",
            original["omega"],
            original["added_mass"],
            original["radiation_damping"],
            extended["omega"],
            extended["added_mass"],
            extended["radiation_damping"],
        ),
        plot_excitation_force_extrapolation_comparison(
            figures_dir / "excitation_force_extrapolation_comparison.png",
            original["omega"],
            original["wave_force"],
            extended["omega"],
            extended["wave_force"],
        ),
    ]

    time_domain_comparison = None
    if args.run_time_domain_comparison:
        time_domain_comparison = run_time_domain_pair(
            data_root=data_root,
            original_hydro_file=args.hydro_file,
            extended_hydro_file=extended_hydro_path,
            output_root=output_root,
            args=args,
        )

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_hydrodynamic_dataset": hydro_path,
        "extrapolated_hydrodynamic_dataset": extended_hydro_path,
        "adapter_layer_only": True,
        "rodm_frequency_core_modified": False,
        "extrapolation_config": config.to_dict(),
        "selected_omega_rad_s": selected_omega,
        "frequency_grid_before": frequency_grid_diagnostics(
            original["omega"],
            reference_omega=selected_omega,
        ),
        "frequency_grid_after": frequency_grid_diagnostics(
            extended["omega"],
            reference_omega=selected_omega,
        ),
        "hydrodynamic_array_diagnostics_before": hydrodynamic_array_diagnostics(
            original["omega"],
            original["added_mass"],
            original["radiation_damping"],
            original["wave_force"],
        ),
        "hydrodynamic_array_diagnostics_after": hydrodynamic_array_diagnostics(
            extended["omega"],
            extended["added_mass"],
            extended["radiation_damping"],
            extended["wave_force"],
        ),
        "max_abs_difference_inside_original_range": {
            "dataset_file_variables": file_invariance,
            "merged_arrays": array_invariance,
        },
        "radiation_kernel_comparison": kernel_comparison,
        "figures": figures,
        "time_domain_comparison": time_domain_comparison,
    }
    metrics_path = write_metrics_json(output_root / "hydrodynamic_extrapolation_metrics.json", metrics)

    print("Hydrodynamic extrapolation validation completed.")
    print(f"extended_hydrodynamic_dataset: {extended_hydro_path}")
    print(f"added_mass_original_range_delta: {array_invariance['added_mass']:.6g}")
    print(f"radiation_damping_original_range_delta: {array_invariance['radiation_damping']:.6g}")
    print(
        "kernel_tail_rms_after_over_before: "
        f"{kernel_comparison['tail_rms_ratio_after_over_before']:.6g}"
    )
    if time_domain_comparison is not None:
        centerline = time_domain_comparison["comparisons"]["centerline_heave_rms_closure_error"]
        print(
            "centerline_heave_rms_closure_error before/after: "
            f"{centerline['before']:.6g} / {centerline['after']:.6g}"
        )
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
