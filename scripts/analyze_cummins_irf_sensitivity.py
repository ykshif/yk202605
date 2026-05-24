"""Analyze Cummins IRF reconstruction sensitivity for a multi-frequency RODM dataset."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    radiation_coefficients_from_discrete_irf,
    radiation_coefficients_from_irf,
    radiation_frequency_window_weights,
)
from offshore_energy_sim.time_domain.rodm_hydrodynamics import (  # noqa: E402
    _reduced_matrix_series,
    prepare_rodm_time_domain_hydrodynamic_terms,
)

from run_time_domain_reference_case_300 import (  # noqa: E402
    build_default_case,
    default_dm_fem_root,
)


DEFAULT_HYDRO = (
    Path("HydrodynamicData")
    / "Yoga"
    / "DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh4.nc"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "time_domain" / "cummins_irf_sensitivity_dm10_mesh4"
BENCHMARK_300M_OMEGA = 0.4157


@dataclass(frozen=True)
class WindowCase:
    case_id: str
    window: str
    start_omega: float | None = None
    stop_omega: float | None = None


WINDOW_CASES = (
    WindowCase("none", "none"),
    WindowCase("cosine_tail_default", "cosine_tail"),
    WindowCase("linear_tail_default", "linear_tail"),
    WindowCase("cosine_tail_from_1p0", "cosine_tail", 1.0, None),
    WindowCase("linear_tail_from_1p0", "linear_tail", 1.0, None),
    WindowCase("linear_full_band", "linear_tail", 0.1, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--hydro-file", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-omega", type=float, default=BENCHMARK_300M_OMEGA)
    parser.add_argument("--frequency-index", type=int, default=None)
    parser.add_argument("--hydro-node-order", choices=("default", "reversed"), default="reversed")
    parser.add_argument("--steps-per-cycle", type=int, default=80)
    parser.add_argument("--memory-cycles", type=float, default=2.0)
    parser.add_argument(
        "--radiation-convolution-rule",
        choices=("rectangular", "trapezoidal"),
        default="rectangular",
    )
    parser.add_argument(
        "--infinite-added-mass-method",
        choices=("high_frequency", "ogilvie"),
        default="high_frequency",
    )
    parser.add_argument(
        "--radiation-passivity-correction",
        choices=("none", "clip_negative_eigenvalues"),
        default="clip_negative_eigenvalues",
    )
    parser.add_argument("--added-mass-tail-count", type=int, default=3)
    return parser.parse_args()


def relative_matrix_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1.0e-30))


def selected_frequency_index(values: np.ndarray, target_omega: float, override: int | None) -> int:
    if override is not None:
        if override < 0 or override >= values.size:
            raise ValueError("--frequency-index is outside the hydrodynamic omega grid")
        return int(override)
    return int(np.argmin(np.abs(values - target_omega)))


def reference_series(case, dataset, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    added = _reduced_matrix_series(dataset["added_mass"].values, case)[order]
    damping = _reduced_matrix_series(dataset["radiation_damping"].values, case)[order]
    return added, damping


def run_window_case(
    window_case: WindowCase,
    *,
    case,
    dataset,
    omega_sorted: np.ndarray,
    target_index_sorted: int,
    added_reference: np.ndarray,
    damping_reference: np.ndarray,
    period: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    config = TimeDomainSimulationConfig(
        time_step=period / args.steps_per_cycle,
        duration=args.memory_cycles * period,
        radiation_model="direct_convolution",
        memory_duration=args.memory_cycles * period,
        infinite_added_mass_method=args.infinite_added_mass_method,
        added_mass_tail_count=args.added_mass_tail_count,
        radiation_passivity_correction=args.radiation_passivity_correction,
        radiation_frequency_window=window_case.window,
        radiation_window_start_omega=window_case.start_omega,
        radiation_window_stop_omega=window_case.stop_omega,
        radiation_convolution_rule=args.radiation_convolution_rule,
    )
    terms = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)
    continuous_added, continuous_damping = radiation_coefficients_from_irf(
        omega_sorted,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
    )
    discrete_added, discrete_damping = radiation_coefficients_from_discrete_irf(
        omega_sorted,
        terms.radiation_irf,
        terms.radiation_irf_time,
        added_mass_infinite=terms.added_mass_infinite,
        convolution_rule=args.radiation_convolution_rule,
    )
    weights = radiation_frequency_window_weights(
        omega_sorted,
        window=window_case.window,
        start_omega=window_case.start_omega,
        stop_omega=window_case.stop_omega,
    )
    irf_norm = np.linalg.norm(terms.radiation_irf.reshape(terms.radiation_irf.shape[0], -1), axis=1)
    return {
        "case_id": window_case.case_id,
        "window": window_case.window,
        "start_omega": window_case.start_omega,
        "stop_omega": window_case.stop_omega,
        "target_weight": float(weights[target_index_sorted]),
        "target_continuous_added_error": relative_matrix_error(
            continuous_added[target_index_sorted],
            added_reference[target_index_sorted],
        ),
        "target_continuous_damping_error": relative_matrix_error(
            continuous_damping[target_index_sorted],
            damping_reference[target_index_sorted],
        ),
        "target_discrete_added_error": relative_matrix_error(
            discrete_added[target_index_sorted],
            added_reference[target_index_sorted],
        ),
        "target_discrete_damping_error": relative_matrix_error(
            discrete_damping[target_index_sorted],
            damping_reference[target_index_sorted],
        ),
        "grid_continuous_added_error": relative_matrix_error(continuous_added, added_reference),
        "grid_continuous_damping_error": relative_matrix_error(continuous_damping, damping_reference),
        "grid_discrete_added_error": relative_matrix_error(discrete_added, added_reference),
        "grid_discrete_damping_error": relative_matrix_error(discrete_damping, damping_reference),
        "irf_norm_initial": float(irf_norm[0]),
        "irf_norm_final": float(irf_norm[-1]),
        "irf_norm_tail_ratio": float(irf_norm[-1] / max(irf_norm[0], 1.0e-30)),
        "irf_time": terms.radiation_irf_time,
        "irf_norm": irf_norm,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        key
        for key in rows[0].keys()
        if key not in {"irf_time", "irf_norm"}
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return path


def plot_error_summary(path: Path, rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["case_id"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    target_b = np.array([float(row["target_discrete_damping_error"]) for row in rows])
    grid_b = np.array([float(row["grid_discrete_damping_error"]) for row in rows])
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - width / 2, target_b, width, label="target omega")
    ax.bar(x + width / 2, grid_b, width, label="all omega grid")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative B reconstruction error")
    ax.set_title("Cummins IRF damping reconstruction sensitivity")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_irf_norms(path: Path, rows: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for row in rows:
        ax.plot(
            row["irf_time"],
            row["irf_norm"],
            linewidth=1.2,
            label=str(row["case_id"]),
        )
    ax.set_xlabel("Memory time (s)")
    ax.set_ylabel("IRF Frobenius norm")
    ax.set_title("Radiation IRF norm by frequency-window option")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    data_root = default_dm_fem_root(args.data_root)
    hydro_path = args.hydro_file
    if not hydro_path.is_absolute():
        hydro_path = data_root / hydro_path
    dataset_for_omega = open_hydrodynamic_dataset(hydro_path, merge_complex=False)
    try:
        omega = np.asarray(dataset_for_omega.omega.values, dtype=float)
    finally:
        dataset_for_omega.close()
    frequency_index = selected_frequency_index(omega, args.target_omega, args.frequency_index)
    selected_omega = float(omega[frequency_index])
    period = 2.0 * np.pi / selected_omega
    order = np.argsort(omega)
    omega_sorted = omega[order]
    target_index_sorted = int(np.where(order == frequency_index)[0][0])

    case = build_default_case(
        data_root,
        reversed_hydro=args.hydro_node_order == "reversed",
        structural_reduction_method="serep_ridge",
    )
    case = replace(
        case,
        case_id=f"irf_sensitivity_{hydro_path.stem}",
        hydrodynamic_dataset=hydro_path,
        frequency_index=frequency_index,
    )

    dataset = open_hydrodynamic_dataset(hydro_path, merge_complex=True)
    try:
        added_ref, damping_ref = reference_series(case, dataset, order)
        rows = [
            run_window_case(
                window_case,
                case=case,
                dataset=dataset,
                omega_sorted=omega_sorted,
                target_index_sorted=target_index_sorted,
                added_reference=added_ref,
                damping_reference=damping_ref,
                period=period,
                args=args,
            )
            for window_case in WINDOW_CASES
        ]
    finally:
        dataset.close()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(output_root / "irf_sensitivity.csv", rows)
    figures = [
        plot_error_summary(output_root / "figures" / "irf_damping_reconstruction_errors.png", rows),
        plot_irf_norms(output_root / "figures" / "irf_norm_by_window.png", rows),
    ]
    serializable_rows = [
        {key: value for key, value in row.items() if key not in {"irf_time", "irf_norm"}}
        for row in rows
    ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hydrodynamic_dataset": str(hydro_path),
        "target_omega_rad_s": args.target_omega,
        "selected_omega_rad_s": selected_omega,
        "frequency_index": frequency_index,
        "steps_per_cycle": args.steps_per_cycle,
        "memory_cycles": args.memory_cycles,
        "radiation_passivity_correction": args.radiation_passivity_correction,
        "radiation_convolution_rule": args.radiation_convolution_rule,
        "rows": serializable_rows,
        "csv": str(csv_path),
        "figures": [str(path) for path in figures],
    }
    json_path = output_root / "irf_sensitivity.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best = min(rows, key=lambda row: float(row["target_discrete_damping_error"]))
    print("Cummins IRF sensitivity completed.")
    print(f"selected_omega_rad_s: {selected_omega:.8g}")
    print(f"best_target_discrete_damping_case: {best['case_id']}")
    print(f"best_target_discrete_damping_error: {best['target_discrete_damping_error']:.6g}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
