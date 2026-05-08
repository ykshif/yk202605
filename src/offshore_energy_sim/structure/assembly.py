"""Structural matrix assembly helpers."""

from __future__ import annotations

import numpy as np


def node_dof_indices(
    node_ids_one_based: list[int] | tuple[int, ...],
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Return global DOF indices for one-based node IDs.

    The matrix convention is node-major with `dofs_per_node` structural DOFs
    per node. For 6-DOF beam/shell nodes this is `[ux, uy, uz, rx, ry, rz]`
    under the current repository convention.
    """

    indices = []
    for node_id in node_ids_one_based:
        start = (node_id - 1) * dofs_per_node
        indices.extend(range(start, start + dofs_per_node))
    return np.array(indices, dtype=int)


def assemble_local_matrix(
    total_nodes: int,
    local_matrix: np.ndarray,
    node_ids_one_based: list[int] | tuple[int, ...],
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Insert a local matrix into a new global matrix.

    This preserves `DM_Assemble.insert_matrix`: a zero global matrix of shape
    `(total_nodes*dofs_per_node, total_nodes*dofs_per_node)` is created and the
    local matrix is added at the requested node DOF positions.
    """

    global_matrix = np.zeros((total_nodes * dofs_per_node, total_nodes * dofs_per_node))
    add_local_matrix_in_place(global_matrix, local_matrix, node_ids_one_based, dofs_per_node)
    return global_matrix


def add_local_matrix_in_place(
    global_matrix: np.ndarray,
    local_matrix: np.ndarray,
    node_ids_one_based: list[int] | tuple[int, ...],
    dofs_per_node: int = 6,
    scale: float = 1.0,
) -> np.ndarray:
    """Add a local matrix to an existing global matrix in place.

    `scale=-1` removes an element stiffness contribution while preserving the
    same DOF mapping.
    """

    indices = node_dof_indices(node_ids_one_based, dofs_per_node)
    for local_row, global_row in enumerate(indices):
        for local_col, global_col in enumerate(indices):
            global_matrix[global_row, global_col] += scale * local_matrix[local_row, local_col]
    return global_matrix


def calculate_node_positions(
    first_node: int,
    node_interval: int,
    num_nodes: int,
) -> list[int]:
    """Return descending one-based control-point node IDs."""

    return [first_node - index * node_interval for index in range(num_nodes)]


def calculate_2d_node_positions_descending(
    first_node: int,
    col_interval: int,
    num_nodes_row: int,
    num_rows: int,
    num_cols: int,
) -> list[int]:
    """Return descending one-based node IDs over a 2D module grid."""

    row_interval = num_nodes_row * col_interval
    nodes = []
    for row in range(num_rows):
        for col in range(num_cols):
            nodes.append(first_node - row * row_interval - col * col_interval)
    return nodes
