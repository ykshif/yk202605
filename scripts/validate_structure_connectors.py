"""Validate refactored structure assembly and hinge connector kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.structure import (  # noqa: E402
    ExplicitHingeSpec,
    HingeLineSpec,
    add_hinge_connections_in_place,
    add_two_node_coupling_in_place,
    apply_explicit_hinge_in_place,
    assemble_local_matrix,
    apply_hinge_line_in_place,
    build_hinged_stiffness,
    calculate_2d_node_positions_descending,
    calculate_column_node_indices,
    calculate_node_positions,
    generate_column_elements,
    hinge_coupling_matrix,
    node_dof_indices,
    remove_element_stiffness_in_place,
)


def assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def validate_node_dof_indices() -> None:
    actual = node_dof_indices([2, 4], dofs_per_node=3)
    expected = np.array([3, 4, 5, 9, 10, 11])
    assert_allclose("node_dof_indices", actual, expected)


def validate_local_matrix_assembly() -> None:
    local = np.arange(36, dtype=float).reshape(6, 6)
    actual = assemble_local_matrix(4, local, [2, 4], dofs_per_node=3)
    expected = np.zeros((12, 12))
    indices = [3, 4, 5, 9, 10, 11]
    for local_row, global_row in enumerate(indices):
        for local_col, global_col in enumerate(indices):
            expected[global_row, global_col] += local[local_row, local_col]
    assert_allclose("assemble_local_matrix", actual, expected)


def validate_node_position_helpers() -> None:
    assert calculate_node_positions(424, 6, 4) == [424, 418, 412, 406]
    assert calculate_2d_node_positions_descending(100, 2, 10, 2, 3) == [
        100,
        98,
        96,
        80,
        78,
        76,
    ]
    assert calculate_column_node_indices(2, nodes_per_row=5, rows_per_column=3) == [2, 7, 12]


def validate_column_elements() -> None:
    actual = generate_column_elements([1, 6, 11], [2, 7, 12])
    expected = [[1, 2, 6, 7], [6, 7, 11, 12]]
    if actual != expected:
        raise AssertionError(f"generate_column_elements mismatch: {actual} != {expected}")


def validate_remove_element_stiffness() -> None:
    global_matrix = np.ones((8, 8)) * 10.0
    element = np.eye(8) * 2.0
    actual = remove_element_stiffness_in_place(
        global_matrix.copy(),
        element,
        elements_one_based=[[1, 2, 3, 4]],
        dofs_per_node=2,
    )
    expected = global_matrix.copy()
    expected -= element
    assert_allclose("remove_element_stiffness_in_place", actual, expected)


def validate_two_node_connector() -> None:
    global_matrix = np.zeros((6, 6))
    coupling = np.diag([5.0, 7.0])
    actual = add_two_node_coupling_in_place(
        global_matrix,
        node_pairs_one_based=[(1, 3)],
        coupling_matrix=coupling,
        dofs_per_node=2,
    )

    expected = np.zeros((6, 6))
    expected[0:2, 0:2] += coupling
    expected[4:6, 4:6] += coupling
    expected[0:2, 4:6] -= coupling
    expected[4:6, 0:2] -= coupling
    assert_allclose("add_two_node_coupling_in_place", actual, expected)


def validate_hinge_connections() -> None:
    coupling = hinge_coupling_matrix(10.0, dofs_per_node=4, released_dofs_zero_based=(2,))
    assert_allclose("hinge_coupling_matrix", coupling, np.diag([10.0, 10.0, 0.0, 10.0]))

    soft_release = hinge_coupling_matrix(
        10.0,
        dofs_per_node=4,
        released_dofs_zero_based=(2,),
        released_dof_stiffness=0.5,
    )
    assert_allclose(
        "hinge_coupling_matrix_soft_release",
        soft_release,
        np.diag([10.0, 10.0, 0.5, 10.0]),
    )

    actual = add_hinge_connections_in_place(
        np.zeros((8, 8)),
        [1],
        [2],
        k_hinge=10.0,
        dofs_per_node=4,
        released_dofs_zero_based=(2,),
    )
    expected = np.zeros((8, 8))
    expected[0:4, 0:4] += coupling
    expected[4:8, 4:8] += coupling
    expected[0:4, 4:8] -= coupling
    expected[4:8, 0:4] -= coupling
    assert_allclose("add_hinge_connections_in_place", actual, expected)

    explicit = ExplicitHingeSpec(
        nodes_side_a_one_based=(1,),
        nodes_side_b_one_based=(2,),
        k_hinge=10.0,
        dofs_per_node=4,
        released_dofs_zero_based=(2,),
        released_dof_stiffness=0.0,
    )
    explicit_actual = apply_explicit_hinge_in_place(np.zeros((8, 8)), explicit)
    assert_allclose("apply_explicit_hinge_in_place", explicit_actual, expected)


def validate_hinge_line_spec() -> None:
    hinge = HingeLineSpec(
        column_a_one_based=2,
        column_b_one_based=3,
        nodes_per_row=5,
        rows_per_column=3,
        k_hinge=10.0,
        dofs_per_node=2,
        released_dofs_zero_based=(1,),
        released_dof_stiffness=0.25,
    )
    if hinge.nodes_side_a_one_based != [2, 7, 12]:
        raise AssertionError(f"unexpected hinge side A nodes: {hinge.nodes_side_a_one_based}")
    if hinge.nodes_side_b_one_based != [3, 8, 13]:
        raise AssertionError(f"unexpected hinge side B nodes: {hinge.nodes_side_b_one_based}")
    if hinge.node_pairs_one_based != [(2, 3), (7, 8), (12, 13)]:
        raise AssertionError(f"unexpected hinge node pairs: {hinge.node_pairs_one_based}")
    if hinge.elements_between_columns_one_based != [[2, 3, 7, 8], [7, 8, 12, 13]]:
        raise AssertionError(f"unexpected hinge elements: {hinge.elements_between_columns_one_based}")

    actual = apply_hinge_line_in_place(np.zeros((26, 26)), hinge)
    expected = add_hinge_connections_in_place(
        np.zeros((26, 26)),
        [2, 7, 12],
        [3, 8, 13],
        k_hinge=10.0,
        dofs_per_node=2,
        released_dofs_zero_based=(1,),
        released_dof_stiffness=0.25,
    )
    assert_allclose("apply_hinge_line_in_place", actual, expected)

    built = build_hinged_stiffness(np.zeros((26, 26)), hinge)
    assert_allclose("build_hinged_stiffness", built, expected)


def main() -> int:
    validations = [
        validate_node_dof_indices,
        validate_local_matrix_assembly,
        validate_node_position_helpers,
        validate_column_elements,
        validate_remove_element_stiffness,
        validate_two_node_connector,
        validate_hinge_connections,
        validate_hinge_line_spec,
    ]
    for validation in validations:
        validation()
        print(f"passed: {validation.__name__}")

    print("Structure connector validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
