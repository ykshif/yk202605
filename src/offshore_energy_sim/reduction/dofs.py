"""Degree-of-freedom indexing helpers for reduced-order models."""

from __future__ import annotations

import numpy as np


def retained_dof_indices(
    num_nodes: int,
    dofs_per_node: int,
    dofs_to_remove_zero_based: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return global DOF indices kept after removing local DOFs per node.

    Global vectors are node-major:
    `[node_1_dof_0, ..., node_1_dof_n, node_2_dof_0, ...]`.
    """

    return np.array(
        [
            index
            for node in range(num_nodes)
            for index in range(node * dofs_per_node, (node + 1) * dofs_per_node)
            if (index - node * dofs_per_node) not in dofs_to_remove_zero_based
        ],
        dtype=int,
    )


def reduce_matrix_dofs(
    matrix: np.ndarray,
    num_nodes: int,
    dofs_to_remove_zero_based: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Remove specified local DOFs from every node of a square matrix.

    This preserves the legacy `SEREP.reduce_dofs` behavior.
    """

    total_dofs = matrix.shape[0]
    dofs_per_node = total_dofs // num_nodes
    keep = retained_dof_indices(num_nodes, dofs_per_node, dofs_to_remove_zero_based)
    return matrix[np.ix_(keep, keep)]


def reduce_force_dofs(
    force: np.ndarray,
    num_nodes: int,
    dof_to_remove_zero_based: int,
) -> np.ndarray:
    """Remove one local DOF from every node of a force vector.

    The input may be 1D or 2D; the legacy behavior flattens it before indexing.
    """

    dofs_per_node = force.size // num_nodes
    keep = [
        index
        for node in range(num_nodes)
        for index in range(node * dofs_per_node, node * dofs_per_node + dofs_per_node)
        if index % dofs_per_node != dof_to_remove_zero_based
    ]
    return force.flatten()[keep]


def separate_master_slave_dofs(
    num_nodes: int,
    master_nodes_one_based: list[int] | tuple[int, ...],
    dofs_per_node: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split retained DOFs into master and slave DOF index arrays."""

    total_dofs = num_nodes * dofs_per_node
    all_dofs = np.arange(total_dofs)

    master_dofs = []
    for node in master_nodes_one_based:
        start = (node - 1) * dofs_per_node
        master_dofs.extend(range(start, start + dofs_per_node))

    slave_dofs = np.delete(all_dofs, master_dofs)
    return np.array(master_dofs), slave_dofs


def reorder_displacement_to_natural_order(
    displacement: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
) -> np.ndarray:
    """Reorder displacement from `[master, slave]` order to natural DOF order.

    This preserves the legacy reverse-master convention used by
    `SEREP.reorder_displacement_matrix`.
    """

    total_dofs = len(master_dofs) + len(slave_dofs)
    current_order = np.concatenate([master_dofs[::-1], slave_dofs])
    natural_order = np.empty(total_dofs, dtype=int)
    natural_order[current_order] = np.arange(total_dofs)
    return displacement[natural_order, :]


def replace_master_dofs_in_global_response(
    master_displacement: np.ndarray,
    global_displacement: np.ndarray,
    control_point_nodes_one_based: list[int] | tuple[int, ...],
    dofs_per_node: int = 5,
) -> np.ndarray:
    """Replace master-node DOFs in a global response vector.

    The control point node order is reversed to preserve the legacy
    `replace_master_with_global` behavior in `DM_Method.py`.
    """

    expanded_indices = []
    for node in control_point_nodes_one_based[::-1]:
        for dof in range(dofs_per_node):
            expanded_indices.append((node - 1) * dofs_per_node + dof)

    for index, global_index in enumerate(expanded_indices):
        global_displacement[global_index] = master_displacement[index]

    return global_displacement
