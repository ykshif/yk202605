"""Module-grid helpers for multi-body hinge/interconnection studies.

The functions in this module standardize the node numbering conventions used
by `RODM_complex_interconnection.py` and `RODM_2D_complex.ipynb` without
changing the numerical hinge kernels. Abaqus node IDs remain one-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from offshore_energy_sim.structure.hinges import ExplicitHingeSpec


ConnectionDirection = Literal["x", "y"]


@dataclass(frozen=True)
class ModuleGridSpec:
    """Square module-grid layout used by the 10x10 hinge study.

    `nodes_per_module_side=7` reproduces the published 30 m x 30 m module
    model: each module has 49 structural nodes and 6 Abaqus DOFs per node
    before the yaw-like sixth DOF is removed for the hydroelastic solve.
    """

    modules_per_side: int = 10
    nodes_per_module_side: int = 7
    module_size: float = 30.0
    dofs_per_node: int = 6
    center_node_one_based: int = 25

    @property
    def module_count(self) -> int:
        """Total number of square modules."""

        return self.modules_per_side * self.modules_per_side

    @property
    def nodes_per_module(self) -> int:
        """Number of structural nodes in one module."""

        return self.nodes_per_module_side * self.nodes_per_module_side

    @property
    def total_nodes(self) -> int:
        """Total structural node count before DOF reduction."""

        return self.module_count * self.nodes_per_module

    @property
    def structure_size(self) -> float:
        """Physical side length of the complete square structure."""

        return self.modules_per_side * self.module_size


@dataclass(frozen=True)
class ModuleControlPoint:
    """One hydrodynamic/control point mapped to a structural master node."""

    point_number: int
    fem_node_number: int
    x: float
    y: float
    module_row: int
    module_column: int


def module_offset_one_based(module_row: int, module_column: int, grid: ModuleGridSpec) -> int:
    """Return the zero-based node offset for a one-based module position."""

    if not 1 <= module_row <= grid.modules_per_side:
        raise ValueError("module_row is outside the grid")
    if not 1 <= module_column <= grid.modules_per_side:
        raise ValueError("module_column is outside the grid")
    module_index_zero_based = (module_row - 1) * grid.modules_per_side + (module_column - 1)
    return module_index_zero_based * grid.nodes_per_module


def generate_module_center_control_points(grid: ModuleGridSpec) -> tuple[ModuleControlPoint, ...]:
    """Generate the center-node master points used by the 10x10 notebook.

    The point order is row-major from the incoming notebook convention:
    top-left module to bottom-right module. The resulting node IDs are
    `25, 74, ..., 4876` for the default 10x10 grid.
    """

    points: list[ModuleControlPoint] = []
    point_number = 1
    half_module = grid.module_size / 2.0
    for row in range(1, grid.modules_per_side + 1):
        y = grid.structure_size - half_module - (row - 1) * grid.module_size
        for column in range(1, grid.modules_per_side + 1):
            x = half_module + (column - 1) * grid.module_size
            fem_node = module_offset_one_based(row, column, grid) + grid.center_node_one_based
            points.append(
                ModuleControlPoint(
                    point_number=point_number,
                    fem_node_number=fem_node,
                    x=x,
                    y=y,
                    module_row=row,
                    module_column=column,
                )
            )
            point_number += 1
    return tuple(points)


def generate_master_nodes_one_based(grid: ModuleGridSpec) -> tuple[int, ...]:
    """Return one-based structural master nodes for hydrodynamic coupling."""

    return tuple(point.fem_node_number for point in generate_module_center_control_points(grid))


def generate_x_hinge_node_pairs(grid: ModuleGridSpec) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return module-edge hinge node lists for horizontal module connections.

    This preserves `generate_hinge_x_pairs(grid_size=10, N=49, nodes_per_row=7,
    total_rows=7)` from the legacy interconnection script.
    """

    hinges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    side = grid.nodes_per_module_side
    for module_row in range(1, grid.modules_per_side + 1):
        for module_column in range(1, grid.modules_per_side):
            left_offset = module_offset_one_based(module_row, module_column, grid)
            right_offset = module_offset_one_based(module_row, module_column + 1, grid)
            side_a = tuple(left_offset + row * side for row in range(1, side + 1))
            side_b = tuple(right_offset + (row - 1) * side + 1 for row in range(1, side + 1))
            hinges.append((side_a, side_b))
    return tuple(hinges)


def generate_y_hinge_node_pairs(grid: ModuleGridSpec) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return module-edge hinge node lists for vertical module connections.

    This preserves `generate_hinge_y_pairs(grid_size=10, N=49, nodes_per_row=7,
    total_rows=7)` from the legacy interconnection script.
    """

    hinges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    side = grid.nodes_per_module_side
    for module_row in range(1, grid.modules_per_side):
        for module_column in range(1, grid.modules_per_side + 1):
            upper_offset = module_offset_one_based(module_row, module_column, grid)
            lower_offset = module_offset_one_based(module_row + 1, module_column, grid)
            side_a = tuple(upper_offset + side * (side - 1) + i for i in range(1, side + 1))
            side_b = tuple(lower_offset + i for i in range(1, side + 1))
            hinges.append((side_a, side_b))
    return tuple(hinges)


def generate_grid_hinge_specs(
    grid: ModuleGridSpec,
    *,
    k_hinge: float = 1.0e10,
    released_dof_stiffness: float = 10.0,
) -> tuple[ExplicitHingeSpec, ...]:
    """Generate all x/y hinge specs for the square module grid.

    The old 10x10 script used different released rotational DOFs by connection
    orientation:
    x direction -> `diag([k, k, k, k, 10, k])`, release local DOF 4.
    y direction -> `diag([k, k, k, 10, k, k])`, release local DOF 3.
    """

    specs: list[ExplicitHingeSpec] = []
    for index, (side_a, side_b) in enumerate(generate_x_hinge_node_pairs(grid), start=1):
        specs.append(
            ExplicitHingeSpec(
                nodes_side_a_one_based=side_a,
                nodes_side_b_one_based=side_b,
                k_hinge=k_hinge,
                dofs_per_node=grid.dofs_per_node,
                released_dofs_zero_based=(4,),
                released_dof_stiffness=released_dof_stiffness,
                name=f"x hinge line {index}",
            )
        )
    for index, (side_a, side_b) in enumerate(generate_y_hinge_node_pairs(grid), start=1):
        specs.append(
            ExplicitHingeSpec(
                nodes_side_a_one_based=side_a,
                nodes_side_b_one_based=side_b,
                k_hinge=k_hinge,
                dofs_per_node=grid.dofs_per_node,
                released_dofs_zero_based=(3,),
                released_dof_stiffness=released_dof_stiffness,
                name=f"y hinge line {index}",
            )
        )
    return tuple(specs)


def drop_duplicate_module_interfaces(grid_values, modules_per_side: int, nodes_per_module_side: int):
    """Remove duplicated interface rows/columns from a tiled module response grid.

    The raw 10x10 structural response grid has shape `(70, 70)`. Adjacent
    modules share physical boundaries, so deleting the last row/column of each
    module except the outer boundary gives a continuous `(61, 61)` field.
    """

    import numpy as np

    delete_indices = [
        nodes_per_module_side * index - 1
        for index in range(1, modules_per_side)
    ]
    merged = np.delete(grid_values, delete_indices, axis=0)
    merged = np.delete(merged, delete_indices, axis=1)
    return merged
