"""Standard 10x10 modular hinge hydroelastic case.

This module extracts the 10x10 case from `RODM_2D_complex.ipynb` and
`RODM_complex_interconnection.py`. The default numerical conventions are kept
notebook-compatible:

* 100 square modules arranged as 10x10;
* one 7x7 structural mesh per 30 m x 30 m module;
* center node of each module retained as the hydrodynamic master node;
* x-direction hinge lines release local DOF 4 with small penalty 10;
* y-direction hinge lines release local DOF 3 with small penalty 10;
* static condensation and `hydrostatic_stiffness / 1.05`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset
from offshore_energy_sim.reduction import (
    reduce_force_dofs,
    retained_dof_indices,
    reorder_displacement_to_natural_order,
    replace_master_dofs_in_global_response,
    separate_master_slave_dofs,
)
from offshore_energy_sim.solver import solve_frequency_domain
from offshore_energy_sim.structure import (
    ExplicitHingeSpec,
    ModuleGridSpec,
    assemble_explicit_hinges_sparse,
    drop_duplicate_module_interfaces,
    generate_grid_hinge_specs,
    generate_master_nodes_one_based,
    read_abaqus_matrix_sparse,
)


ForceOrdering = Literal["none", "reverse_flat", "reverse_nodes"]
MassProjectionOrdering = Literal["legacy_notebook", "master_slave"]


@dataclass(frozen=True)
class ComplexHingeCase:
    """Inputs for the 10x10 complex interconnection benchmark."""

    case_id: str
    title: str
    grid: ModuleGridSpec
    mass_matrix_path: Path
    stiffness_matrix_path: Path
    hydrodynamic_path: Path
    hinges: tuple[ExplicitHingeSpec, ...]
    master_nodes_one_based: tuple[int, ...]
    hydrostatic_divisor: float = 1.05
    frequency_index: int = 0
    hydrodynamic_nodes: int = 100
    retained_dofs_per_node: int = 5
    removed_full_dofs_zero_based: tuple[int, ...] = (5,)
    hydrodynamic_dof_to_remove_zero_based: int = 5
    force_ordering: ForceOrdering = "reverse_flat"
    reverse_master_response_nodes: bool = True
    mass_projection_ordering: MassProjectionOrdering = "legacy_notebook"
    source_programs: tuple[str, ...] = (
        "RODM_2D_complex.ipynb",
        "RODM_complex_interconnection.py",
    )

    @property
    def required_paths(self) -> tuple[Path, ...]:
        """Return required input files for the case."""

        return (
            self.mass_matrix_path,
            self.stiffness_matrix_path,
            self.hydrodynamic_path,
        )


@dataclass(frozen=True)
class ComplexHingeResult:
    """Solved response for the 10x10 complex hinge case."""

    case: ComplexHingeCase
    response: np.ndarray
    heave_grid_raw: np.ndarray
    heave_grid_merged: np.ndarray
    omega: float


def complex_hinge_data_root(default: str | Path = "/Users/yongkang/data/DM-FEM2D") -> Path:
    """Return the DM-FEM2D data root."""

    import os

    return Path(os.environ.get("RODM_DM_FEM_ROOT", default))


def build_complex_hinge_10x10_case(
    data_root: str | Path | None = None,
    *,
    k_hinge: float = 1.0e10,
    released_dof_stiffness: float = 10.0,
) -> ComplexHingeCase:
    """Build the default 10x10 modular hinge case from the old notebook."""

    data_root = complex_hinge_data_root() if data_root is None else Path(data_root)
    grid = ModuleGridSpec(modules_per_side=10, nodes_per_module_side=7, module_size=30.0)
    structure_root = data_root / "StructureData" / "Hinge_complex_paper4"
    hydro_root = data_root / "HydrodynamicData" / "Yoon_hinge"
    hinges = generate_grid_hinge_specs(
        grid,
        k_hinge=k_hinge,
        released_dof_stiffness=released_dof_stiffness,
    )
    return ComplexHingeCase(
        case_id="complex_hinge_10x10_wl180_dir0",
        title="10x10 modular hinge hydroelastic case, wavelength 180 m, direction 0 deg",
        grid=grid,
        mass_matrix_path=structure_root / "Job3030hinge-1_MASS1.mtx",
        stiffness_matrix_path=structure_root / "Job3030hinge-1_STIF1.mtx",
        hydrodynamic_path=hydro_root / "DM10_10_direction0_wl180.nc",
        hinges=hinges,
        master_nodes_one_based=generate_master_nodes_one_based(grid),
    )


def missing_input_paths(case: ComplexHingeCase) -> list[Path]:
    """Return missing input paths for one 10x10 case."""

    return [path for path in case.required_paths if not path.exists()]


def _sparse_block_diagonal_repeat(matrix, count: int):
    """Repeat one module matrix on a sparse block diagonal."""

    from scipy import sparse

    if count < 1:
        raise ValueError("count must be positive")
    return sparse.kron(sparse.eye(count, format="csr"), matrix, format="csr")


def _reduce_sparse_matrix_dofs(matrix, num_nodes: int, dofs_to_remove_zero_based: tuple[int, ...]):
    """Remove local DOFs per node while keeping the matrix sparse."""

    dofs_per_node = matrix.shape[0] // num_nodes
    keep = retained_dof_indices(num_nodes, dofs_per_node, dofs_to_remove_zero_based)
    return matrix[keep][:, keep].tocsr()


def _reverse_node_order_vector(vector: np.ndarray, node_count: int, dofs_per_node: int) -> np.ndarray:
    """Reverse a vector by node blocks while preserving local DOF order."""

    return vector.reshape(node_count, dofs_per_node)[::-1].reshape(node_count * dofs_per_node)


def _static_condensation_sparse(
    stiffness,
    mass,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    mass_projection_ordering: MassProjectionOrdering,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condense sparse structural matrices to dense master matrices.

    The returned `slave_transform` has shape `(n_slave, n_master)` and maps
    master displacement to slave displacement in `[master, slave]` order.
    """

    from scipy.sparse.linalg import splu

    stiffness_mm = stiffness[master_dofs][:, master_dofs]
    stiffness_ms = stiffness[master_dofs][:, slave_dofs]
    stiffness_sm = stiffness[slave_dofs][:, master_dofs]
    stiffness_ss = stiffness[slave_dofs][:, slave_dofs].tocsc()

    solver = splu(stiffness_ss)
    stiffness_ss_inv_sm = solver.solve(stiffness_sm.toarray())
    slave_transform = -stiffness_ss_inv_sm

    reduced_stiffness = stiffness_mm.toarray() - stiffness_ms @ stiffness_ss_inv_sm

    if mass_projection_ordering == "master_slave":
        mass_mm = mass[master_dofs][:, master_dofs]
        mass_ms = mass[master_dofs][:, slave_dofs]
        mass_sm = mass[slave_dofs][:, master_dofs]
        mass_ss = mass[slave_dofs][:, slave_dofs]
        reduced_mass = (
            mass_mm.toarray()
            + mass_ms @ slave_transform
            + slave_transform.T @ mass_sm
            + slave_transform.T @ (mass_ss @ slave_transform)
        )
    else:
        # Notebook-compatible path: legacy SEREP.static_condensation projected
        # the natural-order mass matrix with a [master, slave] transformation.
        transformation = np.vstack([np.eye(len(master_dofs)), slave_transform])
        reduced_mass = transformation.T @ (mass @ transformation)

    return np.asarray(reduced_mass), np.asarray(reduced_stiffness), slave_transform


def _read_hydrodynamic_terms(
    case: ComplexHingeCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and reduce 10x10 Capytaine hydrodynamic terms to 5 DOFs per body."""

    dataset = open_hydrodynamic_dataset(case.hydrodynamic_path, merge_complex=True)
    try:
        omega = float(np.ravel(dataset.omega.values)[case.frequency_index])
        added_mass = dataset["added_mass"][case.frequency_index].values
        radiation_damping = dataset["radiation_damping"][case.frequency_index].values
        hydrostatic_stiffness = dataset["hydrostatic_stiffness"].values
        wave_force = (
            dataset["Froude_Krylov_force"][case.frequency_index].values
            + dataset["diffraction_force"][case.frequency_index].values
        )

        keep = retained_dof_indices(
            case.hydrodynamic_nodes,
            6,
            (case.hydrodynamic_dof_to_remove_zero_based,),
        )
        added_mass = added_mass[np.ix_(keep, keep)]
        radiation_damping = radiation_damping[np.ix_(keep, keep)]
        hydrostatic_stiffness = hydrostatic_stiffness[np.ix_(keep, keep)]
        wave_force = reduce_force_dofs(
            wave_force,
            case.hydrodynamic_nodes,
            case.hydrodynamic_dof_to_remove_zero_based,
        )

        if case.force_ordering == "reverse_flat":
            wave_force = wave_force[::-1]
        elif case.force_ordering == "reverse_nodes":
            wave_force = _reverse_node_order_vector(
                wave_force,
                case.hydrodynamic_nodes,
                case.retained_dofs_per_node,
            )

        return (
            added_mass,
            radiation_damping,
            hydrostatic_stiffness,
            wave_force.reshape(1, -1),
            omega,
        )
    finally:
        dataset.close()


def solve_complex_hinge_case(case: ComplexHingeCase) -> ComplexHingeResult:
    """Solve the 10x10 complex hinge hydroelastic case."""

    missing = missing_input_paths(case)
    if missing:
        raise FileNotFoundError(f"Missing inputs for {case.case_id}: {missing}")

    mass_unit = read_abaqus_matrix_sparse(case.mass_matrix_path)
    stiffness_unit = read_abaqus_matrix_sparse(case.stiffness_matrix_path)
    mass_full = _sparse_block_diagonal_repeat(mass_unit, case.grid.module_count)
    stiffness_full = _sparse_block_diagonal_repeat(stiffness_unit, case.grid.module_count)
    stiffness_full = stiffness_full + assemble_explicit_hinges_sparse(
        case.grid.total_nodes,
        case.hinges,
    )

    # Structural matrices are reduced from 6 DOFs/node to 5 DOFs/node before
    # static condensation, matching the old notebook.
    mass_retained = _reduce_sparse_matrix_dofs(
        mass_full,
        case.grid.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    stiffness_retained = _reduce_sparse_matrix_dofs(
        stiffness_full,
        case.grid.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    master_dofs, slave_dofs = separate_master_slave_dofs(
        case.grid.total_nodes,
        case.master_nodes_one_based,
        dofs_per_node=case.retained_dofs_per_node,
    )
    reduced_mass, reduced_stiffness, slave_transform = _static_condensation_sparse(
        stiffness_retained,
        mass_retained,
        master_dofs,
        slave_dofs,
        mass_projection_ordering=case.mass_projection_ordering,
    )

    added_mass, damping, hydrostatic, wave_force, omega = _read_hydrodynamic_terms(case)
    master_displacement = solve_frequency_domain(
        reduced_mass + added_mass,
        damping,
        reduced_stiffness + hydrostatic / case.hydrostatic_divisor,
        wave_force,
        omega,
    )

    slave_displacement = slave_transform @ master_displacement
    displacement_in_condensed_order = np.vstack([master_displacement, slave_displacement])
    global_response = reorder_displacement_to_natural_order(
        displacement_in_condensed_order,
        master_dofs,
        slave_dofs,
    )

    master_for_replace = master_displacement.reshape(
        len(case.master_nodes_one_based),
        case.retained_dofs_per_node,
    )
    if case.reverse_master_response_nodes:
        master_for_replace = master_for_replace[::-1]
    global_response = replace_master_dofs_in_global_response(
        master_for_replace.reshape(-1, 1),
        global_response,
        case.master_nodes_one_based,
        dofs_per_node=case.retained_dofs_per_node,
    )

    heave_grid_raw = extract_complex_hinge_heave_grid(case, global_response, merge_interfaces=False)
    heave_grid_merged = extract_complex_hinge_heave_grid(case, global_response, merge_interfaces=True)
    return ComplexHingeResult(
        case=case,
        response=global_response,
        heave_grid_raw=heave_grid_raw,
        heave_grid_merged=heave_grid_merged,
        omega=omega,
    )


def extract_complex_hinge_heave_grid(
    case: ComplexHingeCase,
    response: np.ndarray,
    *,
    merge_interfaces: bool = True,
) -> np.ndarray:
    """Extract a 2D heave-response field from the full structural response."""

    heave = np.abs(response[2:: case.retained_dofs_per_node, :]).reshape(-1)
    module_grids = []
    start = 0
    side = case.grid.nodes_per_module_side
    for _module_index in range(case.grid.module_count):
        stop = start + case.grid.nodes_per_module
        module_grids.append(heave[start:stop].reshape(side, side))
        start = stop

    rows = []
    for row_index in range(case.grid.modules_per_side):
        row_modules = module_grids[
            row_index * case.grid.modules_per_side : (row_index + 1) * case.grid.modules_per_side
        ]
        rows.append(np.hstack(row_modules))
    raw_grid = np.vstack(rows)
    if not merge_interfaces:
        return raw_grid
    return drop_duplicate_module_interfaces(
        raw_grid,
        case.grid.modules_per_side,
        case.grid.nodes_per_module_side,
    )


def plot_complex_hinge_result(result: ComplexHingeResult, output_dir: str | Path) -> list[Path]:
    """Write heatmap and centerline figures for the 10x10 case."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grid = result.heave_grid_merged
    extent = [0.0, result.case.grid.structure_size, 0.0, result.case.grid.structure_size]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(grid, origin="upper", extent=extent, cmap="viridis", aspect="equal")
    ax.set_title("10x10 modular hinge heave response")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Heave displacement")
    fig.tight_layout()
    heatmap_path = output_dir / f"{result.case.case_id}_heave_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    paths.append(heatmap_path)

    center_row = grid.shape[0] // 2
    center_column = grid.shape[1] // 2
    x = np.linspace(0.0, 1.0, grid.shape[1])
    y = np.linspace(0.0, 1.0, grid.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(x, grid[center_row, :], color="#d62728", linewidth=1.6, label="center row")
    ax.plot(y, grid[:, center_column], color="#1f77b4", linewidth=1.4, label="center column")
    ax.set_xlabel("normalized coordinate")
    ax.set_ylabel("Heave displacement")
    ax.set_title("10x10 centerline response")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    centerline_path = output_dir / f"{result.case.case_id}_centerlines.png"
    fig.savefig(centerline_path, dpi=300)
    plt.close(fig)
    paths.append(centerline_path)
    return paths
