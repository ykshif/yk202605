"""Run hydrodynamic-isolation RODM comparisons for uniform and non-uniform modules.

Experiment 1:
    Current Capytaine uniform 10 x 1 hydrodynamics, draft 0.5 m,
    compared against the legacy uniform NetCDF files.

Experiment 2:
    Current Capytaine fixed non-uniform 10 x 1 hydrodynamics with smaller
    bow/stern modules and larger middle modules, compared against the same
    legacy uniform NetCDF files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
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
RHO = 1000.0
G = 9.81
CAPYTAINE_N_JOBS = min(16, max(1, (os.cpu_count() or 1) - 2))
FIXED_NONUNIFORM_LENGTHS_M = (20.0, 20.0, 30.0, 40.0, 40.0, 40.0, 40.0, 30.0, 20.0, 20.0)

OUTPUT_ROOT = REPO_ROOT / "results" / "hydrodynamic_isolation_rho1000_20_30_40"
HYDRO_DIR = OUTPUT_ROOT / "hydro"
LEGACY_DM_FEM_ROOT = Path(r"E:\phd\Code\DM-FEM2D")
LEGACY_HYDRO_DIR = LEGACY_DM_FEM_ROOT / "HydrodynamicData" / "Yoga"
STRUCTURE_DIR = LEGACY_DM_FEM_ROOT / "StructureData"
HYDRO_NODE_REVERSE_BY_WAVELENGTH = {300: True}


@dataclass(frozen=True)
class ResponseComparison:
    wavelength_m: int
    omega_rad_s: float
    reference_response_path: Path
    candidate_response_path: Path
    figure_path: Path
    rmse: float
    max_abs_delta: float
    reference_min: float
    reference_max: float
    candidate_min: float
    candidate_max: float
    reference_roughness: float
    candidate_roughness: float


def base_module(mass_kg: float | None = None) -> RectangularModuleSpec:
    """Return the 30 m x 60 m x 2 m module with 0.5 m draft."""

    return RectangularModuleSpec(
        length_m=30.0,
        width_m=60.0,
        height_m=2.0,
        draft_m=DRAFT_M,
        mesh_size_m=2.0,
        vertical_mesh_size_m=2.0,
        mass_kg=mass_kg,
    )


def build_uniform_hydro_config() -> ArrayHydrodynamicsConfig:
    """Current-Capytaine uniform 10 x 1 hydrodynamic dataset."""

    return ArrayHydrodynamicsConfig(
        module=base_module(),
        layout=ArrayLayoutSpec(
            rows=1,
            columns=10,
            spacing_x_m=30.0,
            spacing_y_m=60.0,
            division_mode="uniform",
            total_length_m=300.0,
        ),
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, G),
        output_path=HYDRO_DIR / "current_capytaine_uniform_D0p5_rho1000_wl60_300_mesh2.nc",
        wave_directions_rad=(0.0,),
        water_depth_m=WATER_DEPTH_M,
        rho=RHO,
        g=G,
        n_jobs=CAPYTAINE_N_JOBS,
        compute_rao=False,
    )


def build_fixed_nonuniform_hydro_config() -> ArrayHydrodynamicsConfig:
    """Current-Capytaine fixed non-uniform hydrodynamic dataset."""

    return ArrayHydrodynamicsConfig(
        module=base_module(),
        layout=ArrayLayoutSpec(
            rows=1,
            columns=10,
            spacing_x_m=30.0,
            spacing_y_m=60.0,
            division_mode="custom",
            total_length_m=300.0,
            module_lengths_x_m=FIXED_NONUNIFORM_LENGTHS_M,
        ),
        omegas_rad_s=omega_values_from_wavelengths(WAVELENGTHS_M, WATER_DEPTH_M, G),
        output_path=HYDRO_DIR / "fixed_end_refined_20_30_40_nonuniform_D0p5_rho1000_wl60_300_mesh2.nc",
        wave_directions_rad=(0.0,),
        water_depth_m=WATER_DEPTH_M,
        rho=RHO,
        g=G,
        n_jobs=CAPYTAINE_N_JOBS,
        compute_rao=False,
        structural_grid=StructuralGridSpec(length_m=300.0, width_m=60.0, dx_m=5.0, dy_m=5.0),
    )


def ensure_hydrodynamics(config: ArrayHydrodynamicsConfig, *, force: bool) -> None:
    """Generate a hydrodynamic dataset unless it already exists."""

    if config.output_path.exists() and not force:
        return
    logs: list[str] = []
    start = time.perf_counter()
    run_array_hydrodynamics(config, log=logs.append)
    elapsed = time.perf_counter() - start
    log_path = config.output_path.with_suffix(".generation_log.json")
    log_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def structural_paths() -> StructuralMatrixPaths:
    return StructuralMatrixPaths(
        mass=STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
        stiffness=STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
    )


def legacy_case(wavelength_m: int) -> RodmFrequencyCase:
    """Legacy uniform hydrodynamic RODM case."""

    return RodmFrequencyCase(
        case_id=f"legacy_uniform_{wavelength_m}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=LEGACY_HYDRO_DIR / f"DM10_{wavelength_m}_direction0.nc",
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        frequency_index=0,
        reverse_hydrodynamic_node_order=HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(wavelength_m, False),
    )


def current_uniform_case(wavelength_index: int, hydro_path: Path) -> RodmFrequencyCase:
    """Current Capytaine uniform hydrodynamic RODM case."""

    wavelength_m = WAVELENGTHS_M[wavelength_index]
    return RodmFrequencyCase(
        case_id=f"current_uniform_{wavelength_m}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=hydro_path,
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(wavelength_m, False),
    )


def fixed_nonuniform_case(
    wavelength_index: int,
    hydro_path: Path,
    master_nodes_one_based: tuple[int, ...],
) -> RodmFrequencyCase:
    """Current Capytaine fixed non-uniform hydrodynamic RODM case."""

    wavelength_m = WAVELENGTHS_M[wavelength_index]
    return RodmFrequencyCase(
        case_id=f"fixed_nonuniform_{wavelength_m}m",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        master_nodes_one_based=master_nodes_one_based,
        hydrodynamic_dataset=hydro_path,
        structural_matrices=structural_paths(),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        frequency_index=wavelength_index,
        reverse_hydrodynamic_node_order=HYDRO_NODE_REVERSE_BY_WAVELENGTH.get(wavelength_m, False),
    )


def solve_and_save(case: RodmFrequencyCase, output_path: Path) -> np.ndarray:
    """Solve one RODM case and save the full retained response."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = solve_rodm_frequency_case(case).global_displacement
    np.save(output_path, response)
    return response


def roughness(values: np.ndarray) -> float:
    """Maximum absolute second difference of the centerline heave curve."""

    return float(np.max(np.abs(np.diff(values, 2))))


def compare_responses(
    *,
    wavelength_m: int,
    omega_rad_s: float,
    reference_response: np.ndarray,
    candidate_response: np.ndarray,
    reference_response_path: Path,
    candidate_response_path: Path,
    figure_path: Path,
    title: str,
    candidate_label: str,
) -> ResponseComparison:
    """Compare centerline heave and write an individual figure."""

    import matplotlib.pyplot as plt

    x_reference, heave_reference = extract_centerline_heave(reference_response)
    x_candidate, heave_candidate = extract_centerline_heave(candidate_response)
    delta = heave_candidate - heave_reference

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x_reference, heave_reference, color="#1f77b4", linewidth=1.8, label="legacy uniform NC")
    ax.plot(x_candidate, heave_candidate, color="#d62728", linewidth=1.8, linestyle="--", label=candidate_label)
    ax.set_title(f"{title}\nWavelength {wavelength_m} m")
    ax.set_xlabel("x/L")
    ax.set_ylabel("Heave RAO (m/m)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=260)
    plt.close(fig)

    return ResponseComparison(
        wavelength_m=wavelength_m,
        omega_rad_s=omega_rad_s,
        reference_response_path=reference_response_path,
        candidate_response_path=candidate_response_path,
        figure_path=figure_path,
        rmse=float(np.sqrt(np.mean(delta**2))),
        max_abs_delta=float(np.max(np.abs(delta))),
        reference_min=float(np.min(heave_reference)),
        reference_max=float(np.max(heave_reference)),
        candidate_min=float(np.min(heave_candidate)),
        candidate_max=float(np.max(heave_candidate)),
        reference_roughness=roughness(heave_reference),
        candidate_roughness=roughness(heave_candidate),
    )


def build_panel(
    results: list[ResponseComparison],
    *,
    output_path: Path,
    title: str,
    candidate_label: str,
) -> None:
    """Combine the five response comparisons into a titled panel."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.8))
    axes_flat = axes.ravel()
    for ax, item in zip(axes_flat, results):
        reference = np.load(item.reference_response_path)
        candidate = np.load(item.candidate_response_path)
        x_reference, heave_reference = extract_centerline_heave(reference)
        x_candidate, heave_candidate = extract_centerline_heave(candidate)
        ax.plot(x_reference, heave_reference, color="#1f77b4", linewidth=1.6, label="legacy uniform NC")
        ax.plot(x_candidate, heave_candidate, color="#d62728", linewidth=1.6, linestyle="--", label=candidate_label)
        ax.set_title(f"Wavelength {item.wavelength_m} m")
        ax.set_xlabel("x/L")
        ax.set_ylabel("Heave RAO (m/m)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.85)
        ax.text(
            0.03,
            0.06,
            f"RMSE={item.rmse:.4g}\nmax={item.max_abs_delta:.4g}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.75, "pad": 3},
        )
    for ax in axes_flat[len(results) :]:
        ax.axis("off")
    axes_flat[0].legend(frameon=False, loc="best")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def comparison_rows(results: list[ResponseComparison]) -> list[dict[str, object]]:
    return [
        {
            "wavelength_m": item.wavelength_m,
            "omega_rad_s": item.omega_rad_s,
            "heave_rmse": item.rmse,
            "heave_max_abs_delta": item.max_abs_delta,
            "legacy_heave_min": item.reference_min,
            "legacy_heave_max": item.reference_max,
            "candidate_heave_min": item.candidate_min,
            "candidate_heave_max": item.candidate_max,
            "legacy_roughness": item.reference_roughness,
            "candidate_roughness": item.candidate_roughness,
            "figure_path": str(item.figure_path),
        }
        for item in results
    ]


def write_markdown_table(lines: list[str], results: list[ResponseComparison]) -> None:
    lines.extend(
        [
            "| wavelength m | omega rad/s | heave RMSE | max abs delta | legacy roughness | candidate roughness |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results:
        lines.append(
            f"| {item.wavelength_m} | {item.omega_rad_s:.9g} | "
            f"{item.rmse:.9g} | {item.max_abs_delta:.9g} | "
            f"{item.reference_roughness:.9g} | {item.candidate_roughness:.9g} |"
        )


def run_experiments(*, force_hydro: bool) -> dict[str, object]:
    """Run both isolation experiments and write reports."""

    uniform_config = build_uniform_hydro_config()
    nonuniform_config = build_fixed_nonuniform_hydro_config()
    ensure_hydrodynamics(uniform_config, force=force_hydro)
    ensure_hydrodynamics(nonuniform_config, force=force_hydro)

    mappings = module_structural_node_mappings(nonuniform_config)
    nonuniform_master_nodes = tuple(int(item["fem_node_one_based"]) for item in mappings)
    omegas = tuple(float(value) for value in uniform_config.omegas_rad_s)

    uniform_results: list[ResponseComparison] = []
    nonuniform_results: list[ResponseComparison] = []

    for index, wavelength_m in enumerate(WAVELENGTHS_M):
        legacy_response_path = OUTPUT_ROOT / "legacy_uniform" / f"wavelength_{wavelength_m}m" / "response.npy"
        current_uniform_response_path = OUTPUT_ROOT / "current_uniform" / f"wavelength_{wavelength_m}m" / "response.npy"
        fixed_nonuniform_response_path = OUTPUT_ROOT / "fixed_nonuniform" / f"wavelength_{wavelength_m}m" / "response.npy"

        legacy_response = solve_and_save(legacy_case(wavelength_m), legacy_response_path)
        current_uniform_response = solve_and_save(
            current_uniform_case(index, uniform_config.output_path),
            current_uniform_response_path,
        )
        fixed_nonuniform_response = solve_and_save(
            fixed_nonuniform_case(index, nonuniform_config.output_path, nonuniform_master_nodes),
            fixed_nonuniform_response_path,
        )

        uniform_results.append(
            compare_responses(
                wavelength_m=wavelength_m,
                omega_rad_s=omegas[index],
                reference_response=legacy_response,
                candidate_response=current_uniform_response,
                reference_response_path=legacy_response_path,
                candidate_response_path=current_uniform_response_path,
                figure_path=OUTPUT_ROOT
                / "current_uniform"
                / f"wavelength_{wavelength_m}m"
                / f"legacy_vs_current_uniform_{wavelength_m}m.png",
                title="Experiment 1: Current Capytaine Uniform Modules vs Legacy Uniform NC (rho=1000)",
                candidate_label="current Capytaine uniform NC",
            )
        )
        nonuniform_results.append(
            compare_responses(
                wavelength_m=wavelength_m,
                omega_rad_s=omegas[index],
                reference_response=legacy_response,
                candidate_response=fixed_nonuniform_response,
                reference_response_path=legacy_response_path,
                candidate_response_path=fixed_nonuniform_response_path,
                figure_path=OUTPUT_ROOT
                / "fixed_nonuniform"
                / f"wavelength_{wavelength_m}m"
                / f"legacy_uniform_vs_fixed_nonuniform_{wavelength_m}m.png",
                title="Experiment 2: Fixed 20/30/40 Non-Uniform Modules vs Legacy Uniform NC (rho=1000)",
                candidate_label="fixed end-refined non-uniform NC",
            )
        )

    uniform_panel = OUTPUT_ROOT / "experiment1_current_uniform_vs_legacy_panel.png"
    nonuniform_panel = OUTPUT_ROOT / "experiment2_fixed_nonuniform_vs_legacy_panel.png"
    build_panel(
        uniform_results,
        output_path=uniform_panel,
        title="Experiment 1: Current Capytaine Uniform Modules vs Legacy Uniform NC (rho=1000)",
        candidate_label="current Capytaine uniform NC",
    )
    build_panel(
        nonuniform_results,
        output_path=nonuniform_panel,
        title="Experiment 2: Fixed 20/30/40 Non-Uniform Modules vs Legacy Uniform NC (rho=1000)",
        candidate_label="fixed end-refined non-uniform NC",
    )

    summary = {
        "parameters": {
            "length_m": 300.0,
            "width_m": 60.0,
            "height_m": 2.0,
            "draft_m": DRAFT_M,
            "water_depth_m": WATER_DEPTH_M,
            "rho": RHO,
            "g": G,
            "capytaine_n_jobs": CAPYTAINE_N_JOBS,
            "wavelengths_m": list(WAVELENGTHS_M),
            "fixed_nonuniform_lengths_m": list(FIXED_NONUNIFORM_LENGTHS_M),
            "fixed_nonuniform_boundaries_x_m": list(
                nonuniform_config.layout.x_boundaries(nonuniform_config.module.length_m)
            ),
            "fixed_nonuniform_centers_x_m": [item["x_m"] for item in mappings],
            "fixed_nonuniform_structural_node_ids": list(nonuniform_master_nodes),
        },
        "hydrodynamic_files": {
            "current_uniform": str(uniform_config.output_path),
            "fixed_nonuniform": str(nonuniform_config.output_path),
        },
        "panels": {
            "experiment1": str(uniform_panel),
            "experiment2": str(nonuniform_panel),
        },
        "experiment1_current_uniform_vs_legacy": comparison_rows(uniform_results),
        "experiment2_fixed_nonuniform_vs_legacy": comparison_rows(nonuniform_results),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "isolation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Hydrodynamic Isolation Experiments",
        "",
        "Parameters: 300 m x 60 m x 2 m floating body, draft 0.5 m, water depth 58.5 m, rho 1000 kg/m3.",
        "",
        f"- Capytaine n_jobs: `{CAPYTAINE_N_JOBS}`",
        "",
        f"- current uniform hydrodynamic NC: `{uniform_config.output_path}`",
        f"- fixed non-uniform hydrodynamic NC: `{nonuniform_config.output_path}`",
        f"- fixed non-uniform module lengths m: `{list(FIXED_NONUNIFORM_LENGTHS_M)}`",
        f"- fixed non-uniform x boundaries m: `{list(nonuniform_config.layout.x_boundaries(nonuniform_config.module.length_m))}`",
        f"- fixed non-uniform center x m: `{[item['x_m'] for item in mappings]}`",
        f"- fixed non-uniform structural node ids: `{list(nonuniform_master_nodes)}`",
        "",
        "## Experiment 1: Current Capytaine Uniform Modules vs Legacy Uniform NC (rho=1000)",
        "",
        f"- panel: `{uniform_panel}`",
        "",
    ]
    write_markdown_table(lines, uniform_results)
    lines.extend(
        [
            "",
            "## Experiment 2: Fixed 20/30/40 Non-Uniform Modules vs Legacy Uniform NC (rho=1000)",
            "",
            f"- panel: `{nonuniform_panel}`",
            "",
        ]
    )
    write_markdown_table(lines, nonuniform_results)
    lines.append("")
    (OUTPUT_ROOT / "isolation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-hydro", action="store_true", help="Regenerate current Capytaine hydrodynamic files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [
        *[
            LEGACY_HYDRO_DIR / f"DM10_{wavelength_m}_direction0.nc"
            for wavelength_m in WAVELENGTHS_M
            if not (LEGACY_HYDRO_DIR / f"DM10_{wavelength_m}_direction0.nc").exists()
        ],
        *[
            path
            for path in (
                STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx",
                STRUCTURE_DIR / "JobMesh5_5_STIF1.mtx",
            )
            if not path.exists()
        ],
    ]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(path) for path in missing))

    summary = run_experiments(force_hydro=args.force_hydro)
    print(f"report={OUTPUT_ROOT / 'isolation_report.md'}")
    print(f"summary={OUTPUT_ROOT / 'isolation_summary.json'}")
    print(f"experiment1_panel={summary['panels']['experiment1']}")
    print(f"experiment2_panel={summary['panels']['experiment2']}")
    print("Experiment 1 RMSE/max:")
    for item in summary["experiment1_current_uniform_vs_legacy"]:
        print(
            f"  {item['wavelength_m']} m: rmse={item['heave_rmse']:.6g}, "
            f"max={item['heave_max_abs_delta']:.6g}"
        )
    print("Experiment 2 RMSE/max:")
    for item in summary["experiment2_fixed_nonuniform_vs_legacy"]:
        print(
            f"  {item['wavelength_m']} m: rmse={item['heave_rmse']:.6g}, "
            f"max={item['heave_max_abs_delta']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
