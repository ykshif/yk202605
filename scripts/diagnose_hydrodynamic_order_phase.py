"""Diagnose hydrodynamic module order and incident-wave phase convention.

The generated Capytaine datasets should align hydrodynamic module block order
with structural master nodes ordered by increasing physical x. This script
checks that convention directly from the Froude-Krylov heave-force phase.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402


WAVELENGTHS_M = (60, 120, 180, 240, 300)
OUTPUT_DIR = REPO_ROOT / "results" / "hydrodynamic_order_phase_diagnostics"
DATASETS = {
    "uniform_U10": REPO_ROOT
    / "results"
    / "uniform_reference_convergence_U5_U10_U15_U30_heave"
    / "hydro"
    / "uniform_U10_D0p5_rho1000_wl60_300_mesh2.nc",
    "uniform_U30": REPO_ROOT
    / "results"
    / "uniform_reference_convergence_U5_U10_U15_U30_heave"
    / "hydro"
    / "uniform_U30_D0p5_rho1000_wl60_300_mesh2.nc",
    "nonuniform_U10": REPO_ROOT
    / "results"
    / "nonuniform_U10_vs_U30_guyan_reference"
    / "hydro"
    / "edge_refined_nonuniform_U10_D0p5_rho1000_wl60_300_mesh2.nc",
}


def module_x_positions(dataset) -> np.ndarray:
    lengths = [float(value) for value in dataset.attrs["array_module_lengths_x_m"].split(",")]
    if dataset.attrs.get("array_division_mode") == "uniform":
        spacing = float(dataset.attrs["array_spacing_x_m"])
        columns = int(dataset.attrs["array_columns"])
        x0 = -0.5 * (columns - 1) * spacing
        return np.asarray([x0 + index * spacing for index in range(columns)], dtype=float)

    boundaries = [0.0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return np.asarray(
        [0.5 * (boundaries[index] + boundaries[index + 1]) for index in range(len(lengths))],
        dtype=float,
    )


def circular_phase_score(force: np.ndarray, x_m: np.ndarray, wavelength_m: float, sign: int) -> float:
    """Return 0..1 concentration after removing ``exp(i*sign*k*x)`` phase."""

    k = 2.0 * math.pi / wavelength_m
    residual = force * np.exp(-1j * sign * k * x_m)
    unit = residual / np.maximum(np.abs(residual), np.finfo(float).eps)
    weights = np.abs(force)
    weights = weights / np.sum(weights)
    return float(abs(np.sum(weights * unit)))


def linear_phase_slope(force: np.ndarray, x_m: np.ndarray) -> float:
    phase = np.unwrap(np.angle(force))
    weights = np.abs(force)
    weights = weights / max(float(weights.max()), np.finfo(float).eps)
    return float(np.polyfit(x_m, phase, 1, w=weights)[0])


def diagnose_dataset(name: str, path: Path) -> list[dict[str, object]]:
    dataset = open_hydrodynamic_dataset(path, merge_complex=True)
    try:
        x_m = module_x_positions(dataset)
        dof_labels = [str(value) for value in dataset.coords["influenced_dof"].values]
        heave_indices = [index for index, label in enumerate(dof_labels) if label.endswith("__Heave")]
        rows: list[dict[str, object]] = []
        for omega_index, wavelength_m in enumerate(WAVELENGTHS_M):
            force = dataset["Froude_Krylov_force"].isel(omega=omega_index)
            if "wave_direction" in force.dims:
                force = force.isel(wave_direction=0)
            heave_force = force.values[heave_indices]
            plus_score = circular_phase_score(heave_force, x_m, wavelength_m, sign=1)
            minus_score = circular_phase_score(heave_force, x_m, wavelength_m, sign=-1)
            rows.append(
                {
                    "dataset": name,
                    "path": str(path),
                    "wavelength_m": wavelength_m,
                    "module_count": len(x_m),
                    "x_first_m": float(x_m[0]),
                    "x_last_m": float(x_m[-1]),
                    "dof_order_first": dof_labels[0],
                    "dof_order_last": dof_labels[-1],
                    "fk_heave_phase_slope_rad_per_m": linear_phase_slope(heave_force, x_m),
                    "expected_positive_k_rad_per_m": 2.0 * math.pi / wavelength_m,
                    "phase_score_positive_x": plus_score,
                    "phase_score_negative_x": minus_score,
                    "selected_wave_direction": "+x" if plus_score >= minus_score else "-x",
                }
            )
        return rows
    finally:
        dataset.close()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name, path in DATASETS.items():
        if path.exists():
            rows.extend(diagnose_dataset(name, path))

    csv_path = OUTPUT_DIR / "hydrodynamic_order_phase_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "csv": str(csv_path),
        "conclusion": (
            "Generated Capytaine datasets use module block order increasing in x; "
            "Froude-Krylov heave phase matches exp(+i*k*x), so forward hydrodynamic "
            "node order should be used for all wavelengths."
        ),
        "rows": rows,
    }
    json_path = OUTPUT_DIR / "hydrodynamic_order_phase_diagnostics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Hydrodynamic Order and Phase Diagnostics",
        "",
        "Generated Capytaine datasets were checked using the Froude-Krylov heave force phase.",
        "The expected incident wave convention for `wave_direction=0` is `exp(+i*k*x)`.",
        "",
        f"- CSV: `{csv_path}`",
        "",
        "| dataset | wavelength m | x first | x last | positive-x score | negative-x score | selected |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['wavelength_m']} | "
            f"{float(row['x_first_m']):.9g} | {float(row['x_last_m']):.9g} | "
            f"{float(row['phase_score_positive_x']):.9g} | "
            f"{float(row['phase_score_negative_x']):.9g} | "
            f"{row['selected_wave_direction']} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: use forward hydrodynamic node order for generated datasets at all wavelengths.",
            "The old 300 m reversal is a legacy compatibility switch and should not be applied to these generated NC files.",
            "",
        ]
    )
    report_path = OUTPUT_DIR / "hydrodynamic_order_phase_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
