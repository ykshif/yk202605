"""Compare node-aligned non-uniform module hydrodynamics against uniform RODM cases.

This script keeps the structural model and RODM solver unchanged.  It replaces
only the hydrodynamic NetCDF dataset with a 1D non-uniform, FEM-node-aligned
module layout, then compares centerline heave RAO against the existing uniform
module datasets for the regular-wave wavelengths 60, 120, 180, 240, and 300 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import (  # noqa: E402
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)
from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    module_structural_node_mappings,
    omega_values_from_wavelengths,
    run_array_hydrodynamics,
)
from offshore_energy_sim.postprocess.reference_case_300 import extract_centerline_heave  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


WAVELENGTHS_M = (60, 120, 180, 240, 300)
WATER_DEPTH_M = 58.5
DRAFT_M = 0.5
MODULE_LENGTHS_M = (20.0, 40.0, 30.0, 30.0, 20.0, 40.0, 20.0, 40.0, 30.0, 30.0)
OUTPUT_ROOT = REPO_ROOT / "results" / "nonuniform_module_rodm_comparison"
HYDRO_OUTPUT = OUTPUT_ROOT / "hydro" / "bounded_nonuniform_node_aligned_D0p5_wl60_300_mesh2.nc"
DM_FEM_ROOT = Path(r"E:\phd\Code\DM-FEM2D")
STRUCTURE_DIR = DM_FEM_ROOT / "StructureData"
UNIFORM_HYDRO_DIR = DM_FEM_ROOT / "HydrodynamicData" / "Yoga"
HYDRO_NODE_REVERSE_BY_WAVELENGTH = {300: True}


@dataclass(frozen=True)
class WavelengthResult:
    wavelength_m: int
    omega_rad_s: float
    uniform_response_path: Path
    nonuniform_response_path: Path
    figure_path: Path
    uniform_elapsed_s: float
    nonuniform_elapsed_s: float
    heave_rmse: float
    heave_max_abs_delta: float
    uniform_heave_min: float
    uniform_heave_max: float
    nonuniform_heave_min: float
    nonuniform_heave_max: float


def build_nonuniform_hydro_config() -> ArrayHydrodynamicsConfig:
    """Return the multi-wavelength non-uniform hydrodynamic input deck."""

    module = RectangularModuleSpec(
        length_m=30.0,
        width_m=60.0,
        height_m=2.0,
        draft_m=DRAFT_M,
        mesh_size_m=2.0,
        vertical_mesh_size_m=2.0,
    )
    layout = ArrayLayoutSpec(
        rows=1,
        columns=10,
        spacing_x_m=30.0,
        spacing_y_m=60.0,
        division_mode="random",
        total_length_m=300.0,
        module_lengths_x_m=MODULE_LENGTHS_M,
    )
    return ArrayHydrodynamicsConfig(
        module=module,
        layout=layout,
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, 9.81),
        output_path=HYDRO_OUTPUT,
        wave_directions_rad=(0.0,),
        water_depth_m=WATER_DEPTH_M,
        rho=1025.0,
        g=9.81,
        n_jobs=1,
        compute_rao=False,
        structural_grid=StructuralGridSpec(length_m=300.0, width_m=60.0, dx_m=5.0, dy_m=5.0),
    )


def ensure_nonuniform_hydrodynamics(*, force: bool) -> ArrayHydrodynamicsConfig:
    """Generate or reuse the non-uniform hydrodynamic NetCDF file."""

    config = build_nonuniform_hydro_config()
    if HYDRO_OUTPUT.exists() and not force:
        return config

    logs: list[str] = []
    start = time.perf_counter()
    run_array_hydrodynamics(config, log=logs.append)
    elapsed = time.perf_counter() - start
    log_path = OUTPUT_ROOT / "hydro" / "generation_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config


def build_uniform_case(wavelength_m: int) -> RodmFrequencyCase:
    """Build the existing uniform-module RODM case for one wavelength."""

    return RodmFrequencyCase(
        case_id=f"uniform_{wavelength_m}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=UNIFORM_HYDRO_DIR / f"DM10_{wavelength_m}_direction0.nc",
        structural_matrices=StructuralMatrixPaths(
            mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
            stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
        ),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        frequency_index=0,
        reverse_hydrodynamic_node_order=HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(wavelength_m, False),
    )


def build_nonuniform_case(wavelength_index: int, master_nodes: tuple[int, ...]) -> RodmFrequencyCase:
    """Build the non-uniform-module RODM case for one frequency index."""

    return RodmFrequencyCase(
        case_id=f"nonuniform_{WAVELENGTHS_M[wavelength_index]}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        master_nodes_one_based=master_nodes,
        hydrodynamic_dataset=HYDRO_OUTPUT,
        structural_matrices=StructuralMatrixPaths(
            mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
            stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
        ),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=False,
    )


def solve_and_save(case: RodmFrequencyCase, path: Path) -> tuple[np.ndarray, float]:
    """Run one RODM case and save the retained full-structure response."""

    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = solve_rodm_frequency_case(case)
    elapsed = time.perf_counter() - start
    np.save(path, result.global_displacement)
    return result.global_displacement, elapsed


def plot_comparison(
    *,
    wavelength_m: int,
    uniform_response: np.ndarray,
    nonuniform_response: np.ndarray,
    output_path: Path,
) -> None:
    """Plot uniform vs non-uniform centerline heave RAO."""

    import matplotlib.pyplot as plt

    x_uniform, heave_uniform = extract_centerline_heave(uniform_response)
    x_nonuniform, heave_nonuniform = extract_centerline_heave(nonuniform_response)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_uniform, heave_uniform, color="#1f77b4", linewidth=1.8, label="uniform hydrodynamic modules")
    ax.plot(x_nonuniform, heave_nonuniform, color="#d62728", linewidth=1.8, linestyle="--", label="node-aligned non-uniform modules")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_title(f"RODM hydroelastic response, wavelength {wavelength_m} m")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def compare_one_wavelength(
    wavelength_index: int,
    *,
    master_nodes: tuple[int, ...],
    omegas: tuple[float, ...],
) -> WavelengthResult:
    """Run uniform and non-uniform RODM for one wavelength and compare heave."""

    wavelength_m = WAVELENGTHS_M[wavelength_index]
    uniform_path = OUTPUT_ROOT / f"wavelength_{wavelength_m}m" / "uniform_response.npy"
    nonuniform_path = OUTPUT_ROOT / f"wavelength_{wavelength_m}m" / "nonuniform_response.npy"
    figure_path = OUTPUT_ROOT / f"wavelength_{wavelength_m}m" / f"heave_uniform_vs_nonuniform_{wavelength_m}m.png"

    uniform_response, uniform_elapsed = solve_and_save(build_uniform_case(wavelength_m), uniform_path)
    nonuniform_response, nonuniform_elapsed = solve_and_save(
        build_nonuniform_case(wavelength_index, master_nodes),
        nonuniform_path,
    )

    _, heave_uniform = extract_centerline_heave(uniform_response)
    _, heave_nonuniform = extract_centerline_heave(nonuniform_response)
    delta = heave_nonuniform - heave_uniform
    plot_comparison(
        wavelength_m=wavelength_m,
        uniform_response=uniform_response,
        nonuniform_response=nonuniform_response,
        output_path=figure_path,
    )

    return WavelengthResult(
        wavelength_m=wavelength_m,
        omega_rad_s=float(omegas[wavelength_index]),
        uniform_response_path=uniform_path,
        nonuniform_response_path=nonuniform_path,
        figure_path=figure_path,
        uniform_elapsed_s=uniform_elapsed,
        nonuniform_elapsed_s=nonuniform_elapsed,
        heave_rmse=float(np.sqrt(np.mean(delta**2))),
        heave_max_abs_delta=float(np.max(np.abs(delta))),
        uniform_heave_min=float(np.min(heave_uniform)),
        uniform_heave_max=float(np.max(heave_uniform)),
        nonuniform_heave_min=float(np.min(heave_nonuniform)),
        nonuniform_heave_max=float(np.max(heave_nonuniform)),
    )


def write_summary(config: ArrayHydrodynamicsConfig, results: list[WavelengthResult]) -> None:
    """Write JSON and Markdown comparison summaries."""

    mappings = module_structural_node_mappings(config)
    panel_path = build_summary_panel(results)
    summary = {
        "wavelengths_m": list(WAVELENGTHS_M),
        "water_depth_m": WATER_DEPTH_M,
        "draft_m": DRAFT_M,
        "module_lengths_m": list(MODULE_LENGTHS_M),
        "module_centers_x_m": [item["x_m"] for item in mappings],
        "structural_node_ids": [item["fem_node_one_based"] for item in mappings],
        "nonuniform_hydrodynamic_dataset": str(HYDRO_OUTPUT),
        "summary_panel": str(panel_path),
        "results": [
            {
                "wavelength_m": item.wavelength_m,
                "omega_rad_s": item.omega_rad_s,
                "heave_rmse": item.heave_rmse,
                "heave_max_abs_delta": item.heave_max_abs_delta,
                "uniform_response_path": str(item.uniform_response_path),
                "nonuniform_response_path": str(item.nonuniform_response_path),
                "figure_path": str(item.figure_path),
                "uniform_heave_min": item.uniform_heave_min,
                "uniform_heave_max": item.uniform_heave_max,
                "nonuniform_heave_min": item.nonuniform_heave_min,
                "nonuniform_heave_max": item.nonuniform_heave_max,
            }
            for item in results
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Non-Uniform Module RODM Comparison",
        "",
        "The non-uniform hydrodynamic module centers are aligned to the 5 m structural FEM grid.",
        "",
        f"- hydrodynamic dataset: `{HYDRO_OUTPUT}`",
        f"- draft m: `{DRAFT_M}`",
        f"- module lengths m: `{list(MODULE_LENGTHS_M)}`",
        f"- structural node ids: `{summary['structural_node_ids']}`",
        f"- summary panel: `{panel_path}`",
        "",
        "| wavelength m | omega rad/s | heave RMSE | max abs heave delta | figure |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.wavelength_m} | {item.omega_rad_s:.9g} | "
            f"{item.heave_rmse:.9g} | {item.heave_max_abs_delta:.9g} | "
            f"`{item.figure_path}` |"
        )
    lines.append("")
    (OUTPUT_ROOT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_summary_panel(results: list[WavelengthResult]) -> Path:
    """Combine the five heave comparison plots into one image panel."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    panel_path = OUTPUT_ROOT / "heave_uniform_vs_nonuniform_panel.png"
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.2))
    axes_flat = axes.ravel()
    for ax, item in zip(axes_flat, results):
        ax.imshow(mpimg.imread(item.figure_path))
        ax.set_title(f"Wavelength {item.wavelength_m} m")
        ax.axis("off")
    for ax in axes_flat[len(results) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(panel_path, dpi=220)
    plt.close(fig)
    return panel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate the non-uniform hydrodynamic NetCDF file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [
        UNIFORM_HYDRO_DIR / f"DM10_{wavelength}_direction0.nc"
        for wavelength in WAVELENGTHS_M
        if not (UNIFORM_HYDRO_DIR / f"DM10_{wavelength}_direction0.nc").exists()
    ]
    for path in (STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx", STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx"):
        if not path.exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing required RODM input files: " + ", ".join(str(path) for path in missing))

    config = ensure_nonuniform_hydrodynamics(force=args.force_hydro)
    mappings = module_structural_node_mappings(config)
    master_nodes = tuple(int(item["fem_node_one_based"]) for item in mappings)
    omegas = tuple(float(value) for value in config.omegas_rad_s)

    results = [
        compare_one_wavelength(index, master_nodes=master_nodes, omegas=omegas)
        for index in range(len(WAVELENGTHS_M))
    ]
    write_summary(config, results)

    print(f"nonuniform_hydro={HYDRO_OUTPUT}")
    print(f"module_lengths={list(MODULE_LENGTHS_M)}")
    print(f"structural_node_ids={list(master_nodes)}")
    for item in results:
        print(
            f"{item.wavelength_m} m: heave_rmse={item.heave_rmse:.6g}, "
            f"max_abs_delta={item.heave_max_abs_delta:.6g}, figure={item.figure_path}"
        )
    print(f"summary={OUTPUT_ROOT / 'comparison_summary.json'}")
    print(f"report={OUTPUT_ROOT / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
