"""Load-vector mapping helpers."""

from __future__ import annotations

import numpy as np


def extend_force_vector_to_nodes(
    force_vector: np.ndarray,
    node_ids_one_based: list[int] | tuple[int, ...],
    total_nodes: int,
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Map nodal force blocks into a full global force vector.

    Output shape is `(1, total_nodes * dofs_per_node)`, matching
    `DM_Assemble.extend_force_matrix`.
    """

    extended = np.zeros((1, total_nodes * dofs_per_node), dtype=complex)
    for node_index, node_id in enumerate(node_ids_one_based):
        for dof in range(dofs_per_node):
            extended[0, (node_id - 1) * dofs_per_node + dof] = force_vector[
                0,
                node_index * dofs_per_node + dof,
            ]
    return extended
