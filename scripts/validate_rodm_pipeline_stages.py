"""Validate staged RODM pipeline helpers in the refactor package."""

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
)
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.solver.frequency_domain import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.solver.rodm_frequency import (  # noqa: E402
    prepare_hydrodynamic_terms,
    prepare_structural_reduction,
    solve_rodm_frequency_case,
)
from offshore_energy_sim.structure import calculate_node_positions  # noqa: E402


def _assert_close(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    diff = actual - expected
    max_abs_error = float(np.max(np.abs(diff)))
    l2_relative_error = float(np.linalg.norm(diff) / np.linalg.norm(expected))
    print(label)
    print(f"  actual_shape: {actual.shape}")
    print(f"  expected_shape: {expected.shape}")
    print(f"  max_abs_error: {max_abs_error}")
    print(f"  l2_relative_error: {l2_relative_error}")
    if max_abs_error != 0.0:
        raise AssertionError(f"{label} changed numerical results")


def _solve_from_stages(case):
    master_nodes = calculate_node_positions(
        case.master_node_rule.first_node,
        case.master_node_rule.node_interval,
        case.master_node_rule.count,
    )
    structural = prepare_structural_reduction(case, master_nodes)
    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset, merge_complex=True)
    try:
        hydrodynamic = prepare_hydrodynamic_terms(case, dataset)
    finally:
        dataset.close()

    effective_mass = hydrodynamic.added_mass + structural.reduced_mass
    effective_damping = hydrodynamic.radiation_damping
    effective_stiffness = hydrodynamic.hydrostatic_stiffness + structural.reduced_stiffness
    master_displacement = solve_frequency_domain(
        effective_mass,
        effective_damping,
        effective_stiffness,
        hydrodynamic.wave_force,
        hydrodynamic.omega,
    )
    return reconstruct_global_response(
        structural.transformation,
        master_displacement,
        structural.master_dofs,
        structural.slave_dofs,
    )


def main() -> int:
    paths = default_paths(REPO_ROOT)
    case = build_rodm_frequency_case(paths)
    expected_response = np.load(REPO_ROOT / "results" / "reference_case_300_rodm_generated.npy")

    staged_response = _solve_from_stages(case)
    public_response = solve_rodm_frequency_case(case).global_displacement

    _assert_close("staged_composition_vs_public_solver_default", staged_response, public_response)
    _assert_close("public_solver_vs_pre_refactor_output_default", public_response, expected_response)
    print("RODM staged pipeline validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
