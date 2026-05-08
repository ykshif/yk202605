"""Investigate historical RODM variants against the saved 300 m baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics.netcdf import open_hydrodynamic_dataset  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    build_rodm_frequency_case,
    default_paths,
    extract_centerline_heave,
)
from offshore_energy_sim.reduction import (  # noqa: E402
    reduce_force_dofs,
    reduce_matrix_dofs,
    reorder_displacement_to_natural_order,
    replace_master_dofs_in_global_response,
    separate_master_slave_dofs,
    serep_reduce,
    transform_mass_matrix,
)
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.structure import calculate_node_positions, read_abaqus_matrix_dense  # noqa: E402


def reverse_hydro_node_order_matrix(matrix: np.ndarray, node_count: int = 10, dofs_per_node: int = 5) -> np.ndarray:
    order = np.arange(node_count * dofs_per_node).reshape(node_count, dofs_per_node)[::-1].ravel()
    return matrix[np.ix_(order, order)]


def reverse_hydro_node_order_force(force: np.ndarray, node_count: int = 10, dofs_per_node: int = 5) -> np.ndarray:
    return force.reshape(node_count, dofs_per_node)[::-1].reshape(1, node_count * dofs_per_node)


def metrics(name: str, response: np.ndarray, baseline: np.ndarray) -> dict[str, float | str]:
    diff = response - baseline
    heave = extract_centerline_heave(response)[1]
    heave_baseline = extract_centerline_heave(baseline)[1]
    return {
        "name": name,
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(baseline)),
        "heave_rmse": float(np.sqrt(np.mean((heave - heave_baseline) ** 2))),
        "heave_mean": float(np.mean(heave)),
    }


def main() -> int:
    paths = default_paths(REPO_ROOT)
    case = build_rodm_frequency_case(paths)
    baseline = np.load(paths.response_file)

    master_nodes = calculate_node_positions(
        case.master_node_rule.first_node,
        case.master_node_rule.node_interval,
        case.master_node_rule.count,
    )
    mass = read_abaqus_matrix_dense(case.structural_matrices.mass)
    stiffness = read_abaqus_matrix_dense(case.structural_matrices.stiffness)
    mass = reduce_matrix_dofs(mass, case.total_nodes, case.removed_full_dofs_zero_based)
    stiffness = reduce_matrix_dofs(stiffness, case.total_nodes, case.removed_full_dofs_zero_based)
    mass = transform_mass_matrix(mass, beta=case.mass_blend_beta)

    master_dofs, slave_dofs = separate_master_slave_dofs(case.total_nodes, master_nodes)
    reduced_mass, reduced_stiffness, transform = serep_reduce(
        stiffness,
        mass,
        slave_dofs,
        master_nodes,
    )

    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset)
    try:
        omega = dataset.omega.values
        added_mass = reduce_matrix_dofs(dataset["added_mass"][0].values, 10, [5])
        radiation = reduce_matrix_dofs(dataset["radiation_damping"][0].values, 10, [5])
        hydrostatic = reduce_matrix_dofs(dataset["hydrostatic_stiffness"].values, 10, [5])
        force = reduce_force_dofs(
            dataset["Froude_Krylov_force"][0].values + dataset["diffraction_force"][0].values,
            10,
            5,
        ).reshape(1, 50)
    finally:
        dataset.close()

    variants = {}
    variants["normal"] = (added_mass, radiation, hydrostatic, force)
    variants["force_reversed"] = (added_mass, radiation, hydrostatic, reverse_hydro_node_order_force(force))
    variants["hydro_matrices_and_force_reversed"] = (
        reverse_hydro_node_order_matrix(added_mass),
        reverse_hydro_node_order_matrix(radiation),
        reverse_hydro_node_order_matrix(hydrostatic),
        reverse_hydro_node_order_force(force),
    )

    rows = []
    for name, (a, c, k_h, f) in variants.items():
        master = solve_frequency_domain(
            a + reduced_mass,
            c,
            k_h + reduced_stiffness,
            f,
            omega,
        )
        response = reorder_displacement_to_natural_order(transform @ master, master_dofs, slave_dofs)
        rows.append(metrics(name, response, baseline))

        replaced = replace_master_dofs_in_global_response(
            master[:, 0].copy(),
            response.copy(),
            master_nodes,
        )
        rows.append(metrics(f"{name}_replace_master", replaced, baseline))

    for row in rows:
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
