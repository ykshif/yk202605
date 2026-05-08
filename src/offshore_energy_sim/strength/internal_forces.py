"""Internal-force extraction helpers for module-based structures."""

from __future__ import annotations

import numpy as np


def generate_1d_module_nodes(
    total_nodes: int,
    rows: int,
    module_nodes: int,
    module_number: int,
) -> list[list[int]]:
    """Generate one-based module node IDs for a 1D module distribution."""

    nodes_per_row = total_nodes // rows
    modules = []
    for module_index in range(module_number):
        module = []
        for row in range(rows):
            base = row * nodes_per_row + module_index * module_nodes + 1
            module.extend(list(range(base, base + module_nodes)))
        modules.append(module)

    for module_index in range(1, module_number):
        modules[module_index] = [node - module_index for node in modules[module_index]]

    return modules


def extract_module_displacements(
    displacement: np.ndarray,
    modules_one_based: list[list[int]],
    total_nodes: int,
    dofs_per_node: int,
) -> list[np.ndarray]:
    """Extract module displacement vectors from a global response vector."""

    nodal_displacement = displacement.reshape(total_nodes, dofs_per_node)
    module_displacements = []
    for module in modules_one_based:
        values = nodal_displacement[np.array(module) - 1]
        module_displacements.append(values.reshape(len(module) * dofs_per_node, 1))
    return module_displacements


def compute_module_forces(
    element_stiffness: np.ndarray,
    module_displacements: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute module internal force vectors using `F = K_element * u`."""

    return [element_stiffness @ displacement for displacement in module_displacements]


def map_module_forces_to_global_nodes(
    module_forces: list[np.ndarray],
    modules_one_based: list[list[int]],
    total_nodes: int,
    dofs_per_node: int = 5,
) -> np.ndarray:
    """Map module force vectors back to nodal global force storage."""

    global_forces = np.zeros((total_nodes, dofs_per_node), dtype=np.complex128)
    processed_nodes = set()
    nodes_per_module = len(modules_one_based[0]) if modules_one_based else 0

    for module_index, module in enumerate(modules_one_based):
        node_forces = module_forces[module_index].reshape(nodes_per_module, dofs_per_node)
        for node_index, node in enumerate(module):
            if node not in processed_nodes:
                global_forces[node - 1] = node_forces[node_index]
                processed_nodes.add(node)

    return global_forces


def middle_interface_moment_per_width(
    global_forces: np.ndarray,
    total_nodes: int,
    rows: int,
    module_nodes: int,
    element_width: float,
    dofs_per_node: int = 5,
    moment_dof_zero_based: int = 4,
) -> np.ndarray:
    """Extract middle interface moment per width from global nodal forces."""

    moment = global_forces.reshape(total_nodes * dofs_per_node, 1)[
        moment_dof_zero_based::dofs_per_node
    ]
    start_index = int((rows // 2) * (total_nodes / rows))
    end_index = int(start_index + (total_nodes / rows))
    interval = int(module_nodes - 1)
    return abs(moment[start_index:end_index:interval] / element_width)
