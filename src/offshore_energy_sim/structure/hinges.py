"""Hinge connection helpers for structural stiffness matrices.

The legacy hinge model releases one rotational DOF between two adjacent node
columns and keeps the other relative DOFs stiff. This module keeps the same
matrix operations but exposes them through small, reusable specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from offshore_energy_sim.structure.assembly import add_local_matrix_in_place
from offshore_energy_sim.structure.connectors import add_two_node_coupling_in_place


@dataclass(frozen=True)
class HingeLineSpec:
    """Specification for one straight hinge line between two node columns.

    Node IDs are one-based to match Abaqus matrix files. A 6-DOF node is
    interpreted as `[ux, uy, uz, rx, ry, rz]`; by default local DOF index 4
    (`ry` under the current repository convention) is released. Some published
    hinge notebooks use a tiny nonzero penalty on the released DOF, so the
    released stiffness is configurable while keeping the legacy zero default.
    """

    column_a_one_based: int
    column_b_one_based: int
    nodes_per_row: int
    rows_per_column: int
    k_hinge: float = 1.0e16
    dofs_per_node: int = 6
    released_dofs_zero_based: tuple[int, ...] = (4,)
    released_dof_stiffness: float = 0.0

    @property
    def nodes_side_a_one_based(self) -> list[int]:
        """Return one-based node IDs on the first side of the hinge."""

        return calculate_column_node_indices(
            self.column_a_one_based,
            self.nodes_per_row,
            self.rows_per_column,
        )

    @property
    def nodes_side_b_one_based(self) -> list[int]:
        """Return one-based node IDs on the second side of the hinge."""

        return calculate_column_node_indices(
            self.column_b_one_based,
            self.nodes_per_row,
            self.rows_per_column,
        )

    @property
    def node_pairs_one_based(self) -> list[tuple[int, int]]:
        """Return paired one-based nodes connected by hinge springs."""

        return list(zip(self.nodes_side_a_one_based, self.nodes_side_b_one_based))

    @property
    def elements_between_columns_one_based(self) -> list[list[int]]:
        """Return four-node shell elements spanning the two hinge columns."""

        return generate_column_elements(
            self.nodes_side_a_one_based,
            self.nodes_side_b_one_based,
        )


@dataclass(frozen=True)
class ExplicitHingeSpec:
    """Specification for one hinge connection defined by explicit node pairs.

    This is the standard interface for published Yoon single-/double-hinge
    cases, where the hinge nodes are known after assembling several structural
    submodules. Node IDs are one-based to match Abaqus and notebook inputs.
    """

    nodes_side_a_one_based: tuple[int, ...]
    nodes_side_b_one_based: tuple[int, ...]
    k_hinge: float = 1.0e10
    dofs_per_node: int = 6
    released_dofs_zero_based: tuple[int, ...] = (4,)
    released_dof_stiffness: float = 100.0
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.nodes_side_a_one_based) != len(self.nodes_side_b_one_based):
            raise ValueError("Explicit hinge side node lists must have the same length")

    @property
    def node_pairs_one_based(self) -> list[tuple[int, int]]:
        """Return paired one-based nodes connected by hinge springs."""

        return list(zip(self.nodes_side_a_one_based, self.nodes_side_b_one_based))


def calculate_column_node_indices(
    column_number_one_based: int,
    nodes_per_row: int,
    rows_per_column: int,
) -> list[int]:
    """Return one-based node IDs in a grid column."""

    if column_number_one_based < 1 or column_number_one_based > nodes_per_row:
        raise ValueError("column_number_one_based is outside the grid")

    return [
        (row_index - 1) * nodes_per_row + column_number_one_based
        for row_index in range(1, rows_per_column + 1)
    ]


def generate_column_elements(
    nodes_side_a_one_based: list[int] | tuple[int, ...],
    nodes_side_b_one_based: list[int] | tuple[int, ...],
) -> list[list[int]]:
    """Generate four-node elements between two adjacent node columns."""

    elements = []
    for index in range(len(nodes_side_a_one_based) - 1):
        elements.append(
            [
                nodes_side_a_one_based[index],
                nodes_side_b_one_based[index],
                nodes_side_a_one_based[index + 1],
                nodes_side_b_one_based[index + 1],
            ]
        )
    return elements


def remove_element_stiffness_in_place(
    global_stiffness: np.ndarray,
    element_stiffness: np.ndarray,
    elements_one_based: list[list[int]] | tuple[list[int], ...],
    dofs_per_node: int = 6,
) -> np.ndarray:
    """Subtract element stiffness from a global stiffness matrix.

    This preserves `DM_Hinge.update_global_stiffness_matrix`. For a four-node
    element with 6 DOFs per node, `element_stiffness` has shape `(24, 24)`.
    """

    for element in elements_one_based:
        add_local_matrix_in_place(
            global_stiffness,
            element_stiffness,
            element,
            dofs_per_node=dofs_per_node,
            scale=-1.0,
        )
    return global_stiffness


def hinge_coupling_matrix(
    k_hinge: float,
    dofs_per_node: int = 6,
    released_dofs_zero_based: tuple[int, ...] = (4,),
    released_dof_stiffness: float = 0.0,
) -> np.ndarray:
    """Return the diagonal coupling matrix for a hinge connector.

    The original `DM_Hinge.py` kernel uses `diag([k, k, k, k, 0, k])`, while
    later paper notebooks use small nonzero release penalties such as `1`,
    `10`, or `100`. The default preserves the legacy zero-release behavior.
    """

    diagonal = np.full(dofs_per_node, k_hinge, dtype=float)
    for dof in released_dofs_zero_based:
        diagonal[dof] = released_dof_stiffness
    return np.diag(diagonal)


def add_hinge_connections_in_place(
    global_stiffness: np.ndarray,
    nodes_side_a_one_based: list[int] | tuple[int, ...],
    nodes_side_b_one_based: list[int] | tuple[int, ...],
    k_hinge: float,
    dofs_per_node: int = 6,
    released_dofs_zero_based: tuple[int, ...] = (4,),
    released_dof_stiffness: float = 0.0,
) -> np.ndarray:
    """Add hinge connector stiffness between paired nodes."""

    coupling = hinge_coupling_matrix(
        k_hinge,
        dofs_per_node=dofs_per_node,
        released_dofs_zero_based=released_dofs_zero_based,
        released_dof_stiffness=released_dof_stiffness,
    )
    node_pairs = list(zip(nodes_side_a_one_based, nodes_side_b_one_based))
    return add_two_node_coupling_in_place(
        global_stiffness,
        node_pairs,
        coupling,
        dofs_per_node=dofs_per_node,
    )


def apply_hinge_line_in_place(
    global_stiffness: np.ndarray,
    hinge: HingeLineSpec,
) -> np.ndarray:
    """Add one column-to-column hinge line to a global stiffness matrix.

    `global_stiffness` has shape `(n_nodes*dofs_per_node, n_nodes*dofs_per_node)`.
    The function adds connector penalty terms in place and returns the same
    matrix for chaining.
    """

    return add_hinge_connections_in_place(
        global_stiffness,
        hinge.nodes_side_a_one_based,
        hinge.nodes_side_b_one_based,
        hinge.k_hinge,
        dofs_per_node=hinge.dofs_per_node,
        released_dofs_zero_based=hinge.released_dofs_zero_based,
        released_dof_stiffness=hinge.released_dof_stiffness,
    )


def apply_explicit_hinge_in_place(
    global_stiffness: np.ndarray,
    hinge: ExplicitHingeSpec,
) -> np.ndarray:
    """Add one explicit node-pair hinge specification to a stiffness matrix."""

    return add_hinge_connections_in_place(
        global_stiffness,
        hinge.nodes_side_a_one_based,
        hinge.nodes_side_b_one_based,
        hinge.k_hinge,
        dofs_per_node=hinge.dofs_per_node,
        released_dofs_zero_based=hinge.released_dofs_zero_based,
        released_dof_stiffness=hinge.released_dof_stiffness,
    )


def apply_explicit_hinges_in_place(
    global_stiffness: np.ndarray,
    hinges: list[ExplicitHingeSpec] | tuple[ExplicitHingeSpec, ...],
) -> np.ndarray:
    """Add multiple explicit hinge specifications to a stiffness matrix."""

    for hinge in hinges:
        apply_explicit_hinge_in_place(global_stiffness, hinge)
    return global_stiffness


def assemble_explicit_hinges_sparse(
    total_nodes: int,
    hinges: list[ExplicitHingeSpec] | tuple[ExplicitHingeSpec, ...],
):
    """Assemble explicit hinge stiffness into a SciPy CSR sparse matrix.

    This is the sparse equivalent of repeated `apply_explicit_hinge_in_place`
    calls. Each paired node receives `+KC` on the two diagonal blocks and `-KC`
    on the off-diagonal coupling blocks.
    """

    from scipy.sparse import coo_matrix

    dofs_per_node = hinges[0].dofs_per_node if hinges else 6
    if not hinges:
        size = total_nodes * dofs_per_node
        return coo_matrix((size, size)).tocsr()

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for hinge in hinges:
        if hinge.dofs_per_node != dofs_per_node:
            raise ValueError("All sparse hinge specs must use the same dofs_per_node")
        coupling_diagonal = np.diag(
            hinge_coupling_matrix(
                hinge.k_hinge,
                dofs_per_node=hinge.dofs_per_node,
                released_dofs_zero_based=hinge.released_dofs_zero_based,
                released_dof_stiffness=hinge.released_dof_stiffness,
            )
        )
        for node_a, node_b in hinge.node_pairs_one_based:
            start_a = (node_a - 1) * dofs_per_node
            start_b = (node_b - 1) * dofs_per_node
            for dof, value in enumerate(coupling_diagonal):
                index_a = start_a + dof
                index_b = start_b + dof
                rows.extend((index_a, index_b, index_a, index_b))
                cols.extend((index_a, index_b, index_b, index_a))
                values.extend((value, value, -value, -value))

    size = total_nodes * dofs_per_node
    return coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()


def remove_hinge_line_elements_in_place(
    global_stiffness: np.ndarray,
    element_stiffness: np.ndarray,
    hinge: HingeLineSpec,
) -> np.ndarray:
    """Remove shell element stiffness contributions along one hinge line.

    For four-node shell elements with 6 DOFs per node, `element_stiffness`
    has shape `(24, 24)`. This mirrors the legacy `DM_Hinge` operation used
    when a physical element strip is replaced by hinge connectors.
    """

    return remove_element_stiffness_in_place(
        global_stiffness,
        element_stiffness,
        hinge.elements_between_columns_one_based,
        dofs_per_node=hinge.dofs_per_node,
    )


def build_hinged_stiffness(
    base_stiffness: np.ndarray,
    hinge: HingeLineSpec,
    *,
    element_stiffness_to_remove: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Return stiffness with an optional released element strip and hinge line."""

    stiffness = np.array(base_stiffness, copy=copy)
    if element_stiffness_to_remove is not None:
        remove_hinge_line_elements_in_place(stiffness, element_stiffness_to_remove, hinge)
    apply_hinge_line_in_place(stiffness, hinge)
    return stiffness


def read_symmetric_element_stiffness_matrix(
    path: str | Path,
    matrix_size: int = 24,
) -> np.ndarray:
    """Read one Abaqus element stiffness block stored as an upper triangle."""

    path = Path(path)
    lines = path.read_text().splitlines()

    matrix_data = []
    found_element = False
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if "ELEMENT NUMBER" not in line:
            line_index += 1
            continue

        found_element = True
        while line_index < len(lines) and not lines[line_index].startswith("*MATRIX"):
            line_index += 1
        line_index += 1

        while line_index < len(lines) and not lines[line_index].startswith("** ELEMENT NUMBER"):
            for value in lines[line_index].split(","):
                try:
                    matrix_data.append(float(value.strip().replace("E", "e")))
                except ValueError:
                    pass
            line_index += 1
        break

    if not found_element:
        raise ValueError(f"No element matrix found in {path}")

    matrix = np.zeros((matrix_size, matrix_size))
    data_index = 0
    for row in range(matrix_size):
        for col in range(row, matrix_size):
            matrix[row, col] = matrix_data[data_index]
            matrix[col, row] = matrix_data[data_index]
            data_index += 1
    return matrix


def read_plain_upper_triangle_stiffness_matrix(
    path: str | Path,
    matrix_size: int = 24,
) -> np.ndarray:
    """Read a plain upper-triangle stiffness dump into a symmetric matrix.

    Some Abaqus element stiffness exports in this repository contain only a
    ``*MATRIX,TYPE=STIFFNESS`` header followed by the upper-triangle values,
    without ``** ELEMENT NUMBER`` markers. For a four-node shell element with
    6 DOFs per node, the expected matrix shape is ``(24, 24)``.
    """

    path = Path(path)
    expected_values = matrix_size * (matrix_size + 1) // 2
    tokens = re.findall(
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?",
        path.read_text(),
    )
    values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
    if len(values) < expected_values:
        raise ValueError(
            f"Expected at least {expected_values} numeric values in {path}, "
            f"found {len(values)}"
        )

    matrix = np.zeros((matrix_size, matrix_size))
    data_index = 0
    for row in range(matrix_size):
        for col in range(row, matrix_size):
            matrix[row, col] = values[data_index]
            matrix[col, row] = values[data_index]
            data_index += 1
    return matrix
