"""Validate refactored reduction and frequency-domain solver kernels.

The checks use small deterministic arrays and compare against the legacy
formulas documented in `SEREP.py` and `DM_Assemble.py`. No source data or
baseline result file is modified.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.loads import extend_force_vector_to_nodes  # noqa: E402
from offshore_energy_sim.reduction import (  # noqa: E402
    reduce_force_dofs,
    reduce_matrix_dofs,
    reorder_displacement_to_natural_order,
    replace_master_dofs_in_global_response,
    separate_master_slave_dofs,
    transform_mass_matrix,
)
from offshore_energy_sim.solver import dynamic_stiffness_matrix, solve_frequency_domain  # noqa: E402


def assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def validate_reduce_matrix_dofs() -> None:
    matrix = np.arange(36, dtype=float).reshape(6, 6)
    expected = matrix[np.ix_([0, 2, 3, 5], [0, 2, 3, 5])]
    actual = reduce_matrix_dofs(matrix, num_nodes=2, dofs_to_remove_zero_based=[1])
    assert_allclose("reduce_matrix_dofs", actual, expected)


def validate_reduce_force_dofs() -> None:
    force = np.arange(12, dtype=float).reshape(1, 12)
    expected = force.flatten()[[0, 1, 3, 4, 5, 7, 8, 9, 11]]
    actual = reduce_force_dofs(force, num_nodes=3, dof_to_remove_zero_based=2)
    assert_allclose("reduce_force_dofs", actual, expected)


def validate_mass_transform() -> None:
    mass = np.array([[2.0, 0.5], [0.5, 3.0]])
    lumped = np.diag(mass.sum(axis=1))
    expected = 0.25 * lumped + 0.75 * mass
    actual = transform_mass_matrix(mass, beta=0.25)
    assert_allclose("transform_mass_matrix", actual, expected)


def validate_master_slave_split() -> None:
    master, slave = separate_master_slave_dofs(4, [2, 4], dofs_per_node=2)
    assert_allclose("master_dofs", master, np.array([2, 3, 6, 7]))
    assert_allclose("slave_dofs", slave, np.array([0, 1, 4, 5]))


def validate_reorder_displacement() -> None:
    master = np.array([2, 3])
    slave = np.array([0, 1])
    displacement = np.array([[20.0], [30.0], [0.0], [10.0]])
    expected = np.array([[0.0], [10.0], [30.0], [20.0]])
    actual = reorder_displacement_to_natural_order(displacement, master, slave)
    assert_allclose("reorder_displacement_to_natural_order", actual, expected)


def validate_replace_master() -> None:
    master = np.array([[100.0], [101.0], [200.0], [201.0]])
    global_response = np.zeros((6, 1))
    expected = np.array([[0.0], [0.0], [200.0], [201.0], [100.0], [101.0]])
    actual = replace_master_dofs_in_global_response(
        master,
        global_response,
        control_point_nodes_one_based=[2, 3],
        dofs_per_node=2,
    )
    assert_allclose("replace_master_dofs_in_global_response", actual, expected)


def validate_force_extension() -> None:
    force = np.array([[10.0, 11.0, 20.0, 21.0]], dtype=complex)
    actual = extend_force_vector_to_nodes(force, [3, 1], total_nodes=3, dofs_per_node=2)
    expected = np.array([[20.0, 21.0, 0.0, 0.0, 10.0, 11.0]], dtype=complex)
    assert_allclose("extend_force_vector_to_nodes", actual, expected)


def validate_frequency_solver() -> None:
    mass = np.array([[2.0, 0.1], [0.1, 1.5]])
    damping = np.array([[0.3, 0.0], [0.0, 0.2]])
    stiffness = np.array([[20.0, -2.0], [-2.0, 10.0]])
    force = np.array([[1.0 + 0.5j, 0.2 - 0.1j]])
    omega = 1.3

    response = solve_frequency_domain(mass, damping, stiffness, force, omega)
    residual = dynamic_stiffness_matrix(mass, damping, stiffness, omega) @ response - force.T
    assert_allclose("frequency_solver_residual", residual, np.zeros_like(residual))


def main() -> int:
    validations = [
        validate_reduce_matrix_dofs,
        validate_reduce_force_dofs,
        validate_mass_transform,
        validate_master_slave_split,
        validate_reorder_displacement,
        validate_replace_master,
        validate_force_extension,
        validate_frequency_solver,
    ]
    for validation in validations:
        validation()
        print(f"passed: {validation.__name__}")

    print("Reduction and solver kernel validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
