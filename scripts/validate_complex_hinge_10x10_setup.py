"""Lightweight validation for the 10x10 modular hinge setup."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.validation import build_complex_hinge_10x10_case  # noqa: E402


def main() -> int:
    case = build_complex_hinge_10x10_case()
    x_hinges = [hinge for hinge in case.hinges if hinge.name.startswith("x ")]
    y_hinges = [hinge for hinge in case.hinges if hinge.name.startswith("y ")]

    assert case.grid.module_count == 100
    assert case.grid.total_nodes == 4900
    assert len(case.master_nodes_one_based) == 100
    assert case.master_nodes_one_based[0] == 25
    assert case.master_nodes_one_based[-1] == 4876

    assert len(x_hinges) == 90
    assert len(y_hinges) == 90
    assert sum(len(hinge.node_pairs_one_based) for hinge in case.hinges) == 1260
    assert x_hinges[0].node_pairs_one_based[0] == (7, 50)
    assert x_hinges[0].node_pairs_one_based[-1] == (49, 92)
    assert y_hinges[0].node_pairs_one_based[0] == (43, 491)
    assert y_hinges[0].node_pairs_one_based[-1] == (49, 497)
    assert x_hinges[0].released_dofs_zero_based == (4,)
    assert y_hinges[0].released_dofs_zero_based == (3,)
    assert x_hinges[0].released_dof_stiffness == 10.0

    print("10x10 hinge setup validation passed.")
    print(f"Case: {case.case_id}")
    print(f"Master nodes: {case.master_nodes_one_based[0]} ... {case.master_nodes_one_based[-1]}")
    print(f"Hinge lines: x={len(x_hinges)}, y={len(y_hinges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
