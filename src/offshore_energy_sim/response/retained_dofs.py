"""Utilities for retained-DOF response vectors."""

from __future__ import annotations

import numpy as np


def retained_node_dof_series(
    response: np.ndarray,
    *,
    start_node_one_based: int,
    stop_node_one_based: int,
    retained_dofs_per_node: int,
    dof_index_zero_based: int,
    column: int = 0,
) -> np.ndarray:
    """Extract one DOF over a contiguous one-based node range.

    ``response`` is expected to be arranged as node-major retained DOF blocks:
    ``[node_1_dof_0, ..., node_1_dof_n, node_2_dof_0, ...]``.
    """

    start = start_node_one_based * retained_dofs_per_node - retained_dofs_per_node
    stop = stop_node_one_based * retained_dofs_per_node - retained_dofs_per_node
    node_block = response[start:stop, :]
    return node_block[dof_index_zero_based::retained_dofs_per_node, column]
