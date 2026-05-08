"""Generic structural connector matrix operations."""

from __future__ import annotations

import numpy as np


def add_two_node_coupling_in_place(
    global_matrix: np.ndarray,
    node_pairs_one_based: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    coupling_matrix: np.ndarray,
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Add a two-node connector stiffness to a global stiffness matrix.

    For each pair, the connector contributes `+KC` to each node's diagonal block
    and `-KC` to the off-diagonal coupling blocks:

    `[ +KC  -KC ]`
    `[ -KC  +KC ]`

    The physical meaning is a relative-displacement penalty between paired
    nodes for the DOFs represented by `coupling_matrix`.
    """

    negative_coupling = -coupling_matrix
    for node_a, node_b in node_pairs_one_based:
        index_a = (node_a - 1) * dofs_per_node
        index_b = (node_b - 1) * dofs_per_node

        slice_a = slice(index_a, index_a + dofs_per_node)
        slice_b = slice(index_b, index_b + dofs_per_node)

        global_matrix[slice_a, slice_a] += coupling_matrix
        global_matrix[slice_b, slice_b] += coupling_matrix
        global_matrix[slice_a, slice_b] += negative_coupling
        global_matrix[slice_b, slice_a] += negative_coupling

    return global_matrix
