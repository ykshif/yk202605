"""Simple adapter-layer mooring stiffness helpers."""

from __future__ import annotations

import numpy as np


def corner_node_ids_for_regular_grid(nodes_per_x: int, nodes_per_y: int) -> tuple[int, int, int, int]:
    """Return one-based corner nodes for a row-major rectangular FEM grid."""

    if nodes_per_x < 2 or nodes_per_y < 2:
        raise ValueError("nodes_per_x and nodes_per_y must be at least 2")
    return (
        1,
        int(nodes_per_x),
        int((nodes_per_y - 1) * nodes_per_x + 1),
        int(nodes_per_x * nodes_per_y),
    )


def corner_mooring_diagonal_stiffness(
    *,
    total_nodes: int,
    retained_dofs_per_node: int,
    nodes_per_x: int,
    nodes_per_y: int,
    surge_stiffness: float = 0.0,
    sway_stiffness: float = 0.0,
    heave_stiffness: float = 0.0,
) -> np.ndarray:
    """Return a global retained-DOF diagonal stiffness for four corner springs."""

    if total_nodes <= 0:
        raise ValueError("total_nodes must be positive")
    if retained_dofs_per_node < 3:
        raise ValueError("retained_dofs_per_node must include surge, sway, and heave")
    if nodes_per_x * nodes_per_y != total_nodes:
        raise ValueError("nodes_per_x * nodes_per_y must equal total_nodes")
    stiffness_values = (float(surge_stiffness), float(sway_stiffness), float(heave_stiffness))
    if any(value < 0.0 for value in stiffness_values):
        raise ValueError("mooring stiffness values must be non-negative")

    diagonal = np.zeros(total_nodes * retained_dofs_per_node, dtype=float)
    corner_nodes = corner_node_ids_for_regular_grid(nodes_per_x, nodes_per_y)
    for node in corner_nodes:
        start = (node - 1) * retained_dofs_per_node
        diagonal[start + 0] += stiffness_values[0]
        diagonal[start + 1] += stiffness_values[1]
        diagonal[start + 2] += stiffness_values[2]
    return diagonal


def project_diagonal_stiffness_to_reduced(
    diagonal_stiffness: np.ndarray,
    transformation: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    reverse_master_order: bool = False,
) -> np.ndarray:
    """Project a natural-order global diagonal stiffness into reduced coordinates."""

    diagonal = np.asarray(diagonal_stiffness, dtype=float).reshape(-1)
    transform = np.asarray(transformation, dtype=float)
    master = np.asarray(master_dofs, dtype=int).reshape(-1)
    slave = np.asarray(slave_dofs, dtype=int).reshape(-1)
    if transform.ndim != 2:
        raise ValueError("transformation must be a matrix")
    if diagonal.size != master.size + slave.size:
        raise ValueError("diagonal_stiffness length must match retained global DOFs")
    if transform.shape[0] != diagonal.size or transform.shape[1] != master.size:
        raise ValueError("transformation shape is inconsistent with DOF partitions")
    ordered_master = master[::-1] if reverse_master_order else master
    current_order = np.concatenate([ordered_master, slave])
    diagonal_disordered = diagonal[current_order]
    reduced = transform.T @ (diagonal_disordered[:, np.newaxis] * transform)
    return 0.5 * (reduced + reduced.T)


def build_corner_mooring_reduced_stiffness(
    *,
    total_nodes: int,
    retained_dofs_per_node: int,
    nodes_per_x: int,
    nodes_per_y: int,
    transformation: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    reverse_master_order: bool = False,
    surge_stiffness: float = 0.0,
    sway_stiffness: float = 0.0,
    heave_stiffness: float = 0.0,
) -> np.ndarray:
    """Build and reduce a four-corner spring mooring stiffness matrix."""

    diagonal = corner_mooring_diagonal_stiffness(
        total_nodes=total_nodes,
        retained_dofs_per_node=retained_dofs_per_node,
        nodes_per_x=nodes_per_x,
        nodes_per_y=nodes_per_y,
        surge_stiffness=surge_stiffness,
        sway_stiffness=sway_stiffness,
        heave_stiffness=heave_stiffness,
    )
    return project_diagonal_stiffness_to_reduced(
        diagonal,
        transformation,
        master_dofs,
        slave_dofs,
        reverse_master_order=reverse_master_order,
    )
