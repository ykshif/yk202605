"""Validate RODM case configuration and orchestration imports.

This script intentionally avoids running the full heavy hydroelastic solve when
optional dependencies are unavailable. It verifies that the standardized case
configuration matches the documented 300 m baseline and that the new solver
entry point is importable.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import missing_dependencies  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    build_rodm_frequency_case,
    default_paths,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402
from offshore_energy_sim.structure import calculate_node_positions  # noqa: E402


def main() -> int:
    case = build_rodm_frequency_case(default_paths(REPO_ROOT))
    master_nodes = calculate_node_positions(
        case.master_node_rule.first_node,
        case.master_node_rule.node_interval,
        case.master_node_rule.count,
    )
    expected_master_nodes = [424, 418, 412, 406, 400, 394, 388, 382, 376, 370]

    if master_nodes != expected_master_nodes:
        raise AssertionError(f"master node mismatch: {master_nodes} != {expected_master_nodes}")

    if case.total_nodes * case.retained_dofs_per_node != 3965:
        raise AssertionError("retained full response DOF count should be 3965")

    if case.hydrodynamic_nodes * case.retained_dofs_per_node != 50:
        raise AssertionError("hydrodynamic reduced DOF count should be 50")

    missing = missing_dependencies(("xarray", "capytaine", "scipy"))

    print(f"case_id: {case.case_id}")
    print(f"master_nodes: {master_nodes}")
    print(f"retained_response_dofs: {case.total_nodes * case.retained_dofs_per_node}")
    print(f"hydrodynamic_reduced_dofs: {case.hydrodynamic_nodes * case.retained_dofs_per_node}")
    print(f"solver_entry_point: {solve_rodm_frequency_case.__name__}")
    print(f"missing_optional_dependencies: {missing}")
    print("RODM case orchestration validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
