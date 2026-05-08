"""Read-only structural matrix readers for Abaqus exported matrix files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offshore_energy_sim.utils.hashing import sha256_file


@dataclass(frozen=True)
class AbaqusMatrixSummary:
    """Shape and hash summary for a comma-separated Abaqus matrix export."""

    path: Path
    exists: bool
    sha256: str | None
    dofs_per_node: int
    max_node_id: int | None
    shape: tuple[int, int] | None
    stored_entries: int
    symmetric_entries_estimate: int


def iter_abaqus_matrix_entries(path: str | Path):
    """Yield one-based Abaqus matrix entries: node1, dof1, node2, dof2, value."""

    with Path(path).open("r") as file:
        for line in file:
            fields = line.strip().split(",")
            if len(fields) < 5:
                continue
            yield (
                int(fields[0]),
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                float(fields[4]),
            )


def scan_abaqus_matrix_file(
    path: str | Path,
    *,
    dofs_per_node: int = 6,
) -> AbaqusMatrixSummary:
    """Scan a structural matrix file without materializing the full matrix."""

    path = Path(path)
    if not path.exists():
        return AbaqusMatrixSummary(
            path=path,
            exists=False,
            sha256=None,
            dofs_per_node=dofs_per_node,
            max_node_id=None,
            shape=None,
            stored_entries=0,
            symmetric_entries_estimate=0,
        )

    max_node_id = 0
    stored_entries = 0
    diagonal_entries = 0
    for node1, dof1, node2, dof2, _value in iter_abaqus_matrix_entries(path):
        max_node_id = max(max_node_id, node1, node2)
        stored_entries += 1
        if node1 == node2 and dof1 == dof2:
            diagonal_entries += 1

    size = max_node_id * dofs_per_node
    symmetric_entries = stored_entries * 2 - diagonal_entries
    return AbaqusMatrixSummary(
        path=path,
        exists=True,
        sha256=sha256_file(path),
        dofs_per_node=dofs_per_node,
        max_node_id=max_node_id,
        shape=(size, size),
        stored_entries=stored_entries,
        symmetric_entries_estimate=symmetric_entries,
    )


def read_abaqus_matrix_dense(
    path: str | Path,
    *,
    dofs_per_node: int = 6,
    symmetric: bool = True,
) -> np.ndarray:
    """Read an Abaqus matrix export into a dense NumPy matrix.

    This is equivalent to the legacy `DM_Reading.get_stiffness_matrix` indexing:
    each one-based node has `dofs_per_node` DOFs and file indices are converted
    to zero-based Python row/column indices.
    """

    summary = scan_abaqus_matrix_file(path, dofs_per_node=dofs_per_node)
    if summary.shape is None:
        raise FileNotFoundError(path)

    matrix = np.zeros(summary.shape)
    for node1, dof1, node2, dof2, value in iter_abaqus_matrix_entries(path):
        row = (node1 - 1) * dofs_per_node + dof1 - 1
        col = (node2 - 1) * dofs_per_node + dof2 - 1
        matrix[row, col] = value
        if symmetric and row != col:
            matrix[col, row] = value

    return matrix


def read_abaqus_matrix_sparse(
    path: str | Path,
    *,
    dofs_per_node: int = 6,
    symmetric: bool = True,
):
    """Read an Abaqus matrix export into a SciPy CSR sparse matrix.

    Large modular-grid studies such as the 10x10 hinge model are too large to
    materialize as full dense matrices before reduction. This reader preserves
    the same one-based Abaqus indexing as `read_abaqus_matrix_dense` while
    keeping only nonzero entries.
    """

    from scipy.sparse import coo_matrix

    summary = scan_abaqus_matrix_file(path, dofs_per_node=dofs_per_node)
    if summary.shape is None:
        raise FileNotFoundError(path)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for node1, dof1, node2, dof2, value in iter_abaqus_matrix_entries(path):
        row = (node1 - 1) * dofs_per_node + dof1 - 1
        col = (node2 - 1) * dofs_per_node + dof2 - 1
        rows.append(row)
        cols.append(col)
        values.append(value)
        if symmetric and row != col:
            rows.append(col)
            cols.append(row)
            values.append(value)

    return coo_matrix((values, (rows, cols)), shape=summary.shape).tocsr()
