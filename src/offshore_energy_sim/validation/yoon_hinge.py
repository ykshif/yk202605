"""Standard workflow for Yoon single- and double-hinge validation cases.

This module is the cleaned, scriptable version of
`RODM_Hige_study_plan_a_2.ipynb`. It keeps the verified numerical conventions
from the notebook while removing duplicated cells and hard-coded Windows paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset
from offshore_energy_sim.reduction import (
    reduce_force_dofs,
    reduce_matrix_dofs,
    replace_master_dofs_in_global_response,
    separate_master_slave_dofs,
    serep_reduce,
)
from offshore_energy_sim.response import reconstruct_global_response
from offshore_energy_sim.solver import solve_frequency_domain
from offshore_energy_sim.structure import (
    ExplicitHingeSpec,
    apply_explicit_hinges_in_place,
    read_abaqus_matrix_dense,
)


ReductionMethod = Literal["serep", "static_condensation"]
StaticMassOrdering = Literal["legacy_notebook", "master_slave"]


@dataclass(frozen=True)
class ComparisonLineSpec:
    """One plotted centerline/side-line comparison row."""

    row_index_zero_based: int
    label: str
    reference_curves: tuple[tuple[str, Path], ...] = ()
    experiment_curves: tuple[tuple[str, Path], ...] = ()
    legacy_figure_paths: tuple[Path, ...] = ()
    reverse_model_x: bool = True


@dataclass(frozen=True)
class YoonHingeCase:
    """All inputs needed for one published Yoon hinge validation case."""

    case_id: str
    title: str
    module_count: int
    module_rows: int
    module_columns: int
    total_nodes: int
    mass_matrix_path: Path
    stiffness_matrix_path: Path
    hydrodynamic_path: Path
    hinges: tuple[ExplicitHingeSpec, ...]
    master_nodes_one_based: tuple[int, ...]
    reduction_method: ReductionMethod
    comparison_lines: tuple[ComparisonLineSpec, ...]
    removed_full_dofs_zero_based: tuple[int, ...] = (5,)
    hydrodynamic_dof_to_remove_zero_based: int = 5
    retained_dofs_per_node: int = 5
    hydrodynamic_nodes: int = 10
    frequency_index: int = 0
    hydrostatic_divisor: float = 1.0
    reverse_force_node_order: bool = True
    reverse_master_response_nodes: bool = True
    columns_to_delete_zero_based: tuple[int, ...] = ()
    static_mass_ordering: StaticMassOrdering = "legacy_notebook"

    @property
    def nodes_per_module(self) -> int:
        """Number of structural nodes in one assembled module."""

        return self.module_rows * self.module_columns

    @property
    def required_paths(self) -> tuple[Path, ...]:
        """Return required input paths for solving this case."""

        return (
            self.mass_matrix_path,
            self.stiffness_matrix_path,
            self.hydrodynamic_path,
        )


@dataclass(frozen=True)
class YoonHingeResult:
    """Numerical result for one solved Yoon hinge case."""

    case: YoonHingeCase
    response: np.ndarray
    heave_grid: np.ndarray
    omega: float


def yoon_hinge_data_root(default: str | Path = "/Users/yongkang/data/DM-FEM2D") -> Path:
    """Return the default external DM-FEM2D data root."""

    import os

    return Path(os.environ.get("RODM_DM_FEM_ROOT", default))


def default_reference_root(repo_root: str | Path | None = None) -> Path:
    """Return the local reference archive created outside OneDrive."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return Path(repo_root) / "references" / "hinge_published"


def block_diagonal_repeat(matrix: np.ndarray, count: int) -> np.ndarray:
    """Repeat one structural matrix on a dense block diagonal."""

    if count < 1:
        raise ValueError("count must be positive")
    return np.kron(np.eye(count, dtype=matrix.dtype), matrix)


def missing_input_paths(case: YoonHingeCase) -> list[Path]:
    """Return missing structural/hydrodynamic input files for one case."""

    return [path for path in case.required_paths if not path.exists()]


def _reverse_node_order_vector(
    vector: np.ndarray,
    node_count: int,
    dofs_per_node: int,
) -> np.ndarray:
    """Reverse a reduced force/response vector by node blocks."""

    return vector.reshape(node_count, dofs_per_node)[::-1].reshape(node_count * dofs_per_node)


def _static_condensation_reduce(
    stiffness: np.ndarray,
    mass: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    mass_ordering: StaticMassOrdering,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static-condensation branch matching the verified notebook convention."""

    stiffness_mm = stiffness[np.ix_(master_dofs, master_dofs)]
    stiffness_ms = stiffness[np.ix_(master_dofs, slave_dofs)]
    stiffness_sm = stiffness[np.ix_(slave_dofs, master_dofs)]
    stiffness_ss = stiffness[np.ix_(slave_dofs, slave_dofs)]

    stiffness_ss_inv_sm = np.linalg.solve(stiffness_ss, stiffness_sm)
    reduced_stiffness = stiffness_mm - stiffness_ms @ stiffness_ss_inv_sm
    transformation = np.vstack(
        [
            np.eye(len(master_dofs), dtype=stiffness.dtype),
            -stiffness_ss_inv_sm,
        ]
    )

    if mass_ordering == "master_slave":
        order = np.concatenate([master_dofs, slave_dofs])
        mass_for_projection = mass[np.ix_(order, order)]
    else:
        mass_for_projection = mass
    reduced_mass = transformation.T @ mass_for_projection @ transformation
    return reduced_mass, reduced_stiffness, transformation


def _reduce_structural_matrices(
    case: YoonHingeCase,
    mass_full: np.ndarray,
    stiffness_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply hinges and reduce full structural matrices to master DOFs."""

    stiffness_hinged = np.array(stiffness_full, copy=True)
    apply_explicit_hinges_in_place(stiffness_hinged, case.hinges)

    mass_retained = reduce_matrix_dofs(
        mass_full,
        case.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    stiffness_retained = reduce_matrix_dofs(
        stiffness_hinged,
        case.total_nodes,
        case.removed_full_dofs_zero_based,
    )
    master_dofs, slave_dofs = separate_master_slave_dofs(
        case.total_nodes,
        case.master_nodes_one_based,
        dofs_per_node=case.retained_dofs_per_node,
    )

    if case.reduction_method == "serep":
        reduced_mass, reduced_stiffness, transformation = serep_reduce(
            stiffness_retained,
            mass_retained,
            slave_dofs,
            case.master_nodes_one_based,
            dofs_per_master_node=case.retained_dofs_per_node,
        )
    else:
        reduced_mass, reduced_stiffness, transformation = _static_condensation_reduce(
            stiffness_retained,
            mass_retained,
            master_dofs,
            slave_dofs,
            mass_ordering=case.static_mass_ordering,
        )

    return reduced_mass, reduced_stiffness, transformation, master_dofs, slave_dofs


def _read_hydrodynamic_terms(
    case: YoonHingeCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load and reduce hydrodynamic terms for one Yoon hinge case."""

    dataset = open_hydrodynamic_dataset(case.hydrodynamic_path, merge_complex=True)
    try:
        omega_values = dataset.omega.values
        omega = float(np.ravel(omega_values)[case.frequency_index])
        added_mass = dataset["added_mass"][case.frequency_index].values
        radiation_damping = dataset["radiation_damping"][case.frequency_index].values
        wave_force = (
            dataset["Froude_Krylov_force"][case.frequency_index].values
            + dataset["diffraction_force"][case.frequency_index].values
        )
        hydrostatic_stiffness = dataset["hydrostatic_stiffness"].values

        added_mass = reduce_matrix_dofs(
            added_mass,
            case.hydrodynamic_nodes,
            [case.hydrodynamic_dof_to_remove_zero_based],
        )
        radiation_damping = reduce_matrix_dofs(
            radiation_damping,
            case.hydrodynamic_nodes,
            [case.hydrodynamic_dof_to_remove_zero_based],
        )
        hydrostatic_stiffness = reduce_matrix_dofs(
            hydrostatic_stiffness,
            case.hydrodynamic_nodes,
            [case.hydrodynamic_dof_to_remove_zero_based],
        )
        wave_force = reduce_force_dofs(
            wave_force,
            case.hydrodynamic_nodes,
            case.hydrodynamic_dof_to_remove_zero_based,
        )
        if case.reverse_force_node_order:
            wave_force = _reverse_node_order_vector(
                wave_force,
                case.hydrodynamic_nodes,
                case.retained_dofs_per_node,
            )
        return added_mass, radiation_damping, hydrostatic_stiffness, wave_force.reshape(1, -1), omega
    finally:
        dataset.close()


def solve_yoon_hinge_case(case: YoonHingeCase) -> YoonHingeResult:
    """Solve one Yoon hinge case using the cleaned standard workflow."""

    missing = missing_input_paths(case)
    if missing:
        raise FileNotFoundError(f"Missing inputs for {case.case_id}: {missing}")

    mass_unit = read_abaqus_matrix_dense(case.mass_matrix_path)
    stiffness_unit = read_abaqus_matrix_dense(case.stiffness_matrix_path)
    mass_full = block_diagonal_repeat(mass_unit, case.module_count)
    stiffness_full = block_diagonal_repeat(stiffness_unit, case.module_count)

    reduced_mass, reduced_stiffness, transformation, master_dofs, slave_dofs = (
        _reduce_structural_matrices(case, mass_full, stiffness_full)
    )
    added_mass, damping, hydrostatic, wave_force, omega = _read_hydrodynamic_terms(case)
    master_displacement = solve_frequency_domain(
        reduced_mass + added_mass,
        damping,
        reduced_stiffness + hydrostatic / case.hydrostatic_divisor,
        wave_force,
        omega,
    )
    global_response = reconstruct_global_response(
        transformation,
        master_displacement,
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

    return YoonHingeResult(
        case=case,
        response=global_response,
        heave_grid=extract_yoon_hinge_heave_grid(case, global_response),
        omega=omega,
    )


def extract_yoon_hinge_heave_grid(
    case: YoonHingeCase,
    response: np.ndarray,
) -> np.ndarray:
    """Extract the post-processed heave grid used in the notebook plots."""

    heave = np.abs(response[2:: case.retained_dofs_per_node, :])
    module_values = []
    start = 0
    for _ in range(case.module_count):
        stop = start + case.nodes_per_module
        module_values.append(heave[start:stop].reshape(case.module_rows, case.module_columns))
        start = stop

    combined = np.hstack(module_values)
    if case.columns_to_delete_zero_based:
        combined = np.delete(combined, case.columns_to_delete_zero_based, axis=1)
    return combined


def _load_xy(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a two-column reference curve, returning None when absent."""

    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def plot_yoon_hinge_case(
    result: YoonHingeResult,
    output_dir: str | Path,
) -> list[Path]:
    """Plot all configured comparison rows for one solved hinge case."""

    import matplotlib.pyplot as plt

    case = result.case
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    x_model = np.linspace(0.0, 1.0, result.heave_grid.shape[1])
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.linewidth": 0.9})

    for line in case.comparison_lines:
        y_model = result.heave_grid[line.row_index_zero_based, :]
        if line.reverse_model_x:
            y_model = y_model[::-1]

        fig, ax = plt.subplots(figsize=(7.4, 2.4))
        for label, reference_path in line.reference_curves:
            reference = _load_xy(reference_path)
            if reference is not None:
                ax.plot(reference[0], reference[1], color="#1f1f1f", linewidth=1.4, label=label)

        ax.plot(x_model, y_model, color="#d62728", linewidth=1.6, label="RODM")

        for label, experiment_path in line.experiment_curves:
            experiment = _load_xy(experiment_path)
            if experiment is not None:
                ax.scatter(
                    experiment[0],
                    experiment[1],
                    color="#1f1f1f",
                    s=24,
                    facecolors="none",
                    label=label,
                    zorder=3,
                )

        ax.set_xlabel("x/L")
        ax.set_ylabel("Heave displacement")
        ax.set_title(f"{case.title} - {line.label}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 2.0)
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        output_path = output_dir / f"{case.case_id}_{line.label.replace(' ', '_').lower()}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def _hinge_spec(
    name: str,
    side_a: list[int],
    side_b: list[int],
    *,
    k_hinge: float = 1.0e10,
    released_dof_stiffness: float = 100.0,
) -> ExplicitHingeSpec:
    """Create an explicit hinge spec with tuple-normalized node lists."""

    return ExplicitHingeSpec(
        nodes_side_a_one_based=tuple(side_a),
        nodes_side_b_one_based=tuple(side_b),
        k_hinge=k_hinge,
        released_dof_stiffness=released_dof_stiffness,
        name=name,
    )


def build_yoon_hinge_cases(
    data_root: str | Path | None = None,
    reference_root: str | Path | None = None,
) -> dict[str, YoonHingeCase]:
    """Build the cleaned single-/double-hinge cases from the verified notebook."""

    data_root = yoon_hinge_data_root() if data_root is None else Path(data_root)
    reference_root = default_reference_root() if reference_root is None else Path(reference_root)
    structure_root = data_root / "StructureData" / "Yoon_hinge"
    hydro_root = data_root / "HydrodynamicData" / "Yoon_hinge"
    ref_yoon = reference_root / "csv" / "yoon_numerical"
    ref_hinge = reference_root / "csv" / "fem_reducev2_hinge"
    ref_figures = reference_root / "figures"

    single_hinge = (
        _hinge_spec(
            "single hinge",
            list(range(31, 404, 31)),
            list(range(404, 777, 31)),
        ),
    )
    double_hinges = (
        _hinge_spec(
            "double hinge line 1",
            list(range(21, 274, 21)),
            list(range(274, 527, 21)),
        ),
        _hinge_spec(
            "double hinge line 2",
            list(range(294, 547, 21)),
            list(range(547, 800, 21)),
        ),
    )

    return {
        "single_180": YoonHingeCase(
            case_id="single_180",
            title="Yoon single hinge, 180 deg",
            module_count=2,
            module_rows=13,
            module_columns=31,
            total_nodes=806,
            mass_matrix_path=structure_root / "Job_hinge_study_150_60_YoonModel_MASS1.mtx",
            stiffness_matrix_path=structure_root / "Job_hinge_study_150_60_YoonModel_STIF1.mtx",
            hydrodynamic_path=hydro_root / "DM10_direction180_slender180_rho1025.nc",
            hinges=single_hinge,
            master_nodes_one_based=tuple(sorted([214, 208, 202, 196, 190, 617, 611, 605, 599, 593])),
            reduction_method="serep",
            hydrostatic_divisor=1.02,
            reverse_force_node_order=True,
            columns_to_delete_zero_based=(31,),
            comparison_lines=(
                ComparisonLineSpec(
                    row_index_zero_based=7,
                    label="centerline",
                    # The available CSV files in FEM_Reducev2 do not reproduce
                    # the published single-hinge reference curve reliably.
                    # Keep the current RODM curve clean and compare visually
                    # with the archived paper figure rendered by the runner.
                    reference_curves=(),
                    experiment_curves=(),
                    legacy_figure_paths=(ref_figures / "Yoon-1-hige-180.pdf",),
                    reverse_model_x=True,
                ),
            ),
        ),
        "double_180": YoonHingeCase(
            case_id="double_180",
            title="Yoon double hinge, 180 deg",
            module_count=3,
            module_rows=13,
            module_columns=21,
            total_nodes=819,
            mass_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_MASS1_rho282.mtx",
            stiffness_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_STIF1_rho282.mtx",
            hydrodynamic_path=hydro_root / "DM10_direction180_slender180_rho1025.nc",
            hinges=double_hinges,
            master_nodes_one_based=tuple(
                sorted([130, 136, 142, 401, 407, 413, 419, 678, 684, 690])
            ),
            reduction_method="static_condensation",
            hydrostatic_divisor=1.05,
            reverse_force_node_order=True,
            columns_to_delete_zero_based=(20, 41),
            comparison_lines=(
                ComparisonLineSpec(
                    row_index_zero_based=0,
                    label="case_1",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_0_3.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige-180-180-1.pdf",),
                    reverse_model_x=True,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=7,
                    label="case_2_centerline",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_0_2.csv"),),
                    experiment_curves=(("Experiment", ref_hinge / "Yoon_exp.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige-180-180-2.pdf",),
                    reverse_model_x=True,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=12,
                    label="case_3",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_0_1.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige-180-180-3.pdf",),
                    reverse_model_x=True,
                ),
            ),
        ),
        "double_210": YoonHingeCase(
            case_id="double_210",
            title="Yoon double hinge, 210 deg",
            module_count=3,
            module_rows=13,
            module_columns=21,
            total_nodes=819,
            mass_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_MASS1_rho282.mtx",
            stiffness_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_STIF1_rho282.mtx",
            hydrodynamic_path=hydro_root / "DM10_direction210_slender180_rho1025.nc",
            hinges=double_hinges,
            master_nodes_one_based=tuple(
                sorted([130, 136, 142, 401, 407, 413, 419, 678, 684, 690])
            ),
            reduction_method="serep",
            hydrostatic_divisor=1.05,
            reverse_force_node_order=False,
            columns_to_delete_zero_based=(20, 41),
            comparison_lines=(
                ComparisonLineSpec(
                    row_index_zero_based=0,
                    label="case_1",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_30_1.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-210-1.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=7,
                    label="case_2_centerline",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_30_2.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-210-2.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=12,
                    label="case_3",
                    reference_curves=(("Yoon et al.", ref_yoon / "Yoon_numerical_30_3.csv"),),
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-210-3.pdf",),
                    reverse_model_x=False,
                ),
            ),
        ),
        "double_240": YoonHingeCase(
            case_id="double_240",
            title="Yoon double hinge, 240 deg",
            module_count=3,
            module_rows=13,
            module_columns=21,
            total_nodes=819,
            mass_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_MASS1_rho282.mtx",
            stiffness_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_STIF1_rho282.mtx",
            hydrodynamic_path=hydro_root / "DM10_direction240_slender180_rho1025.nc",
            hinges=double_hinges,
            master_nodes_one_based=tuple(
                sorted([130, 136, 142, 401, 407, 413, 419, 678, 684, 690])
            ),
            reduction_method="serep",
            hydrostatic_divisor=1.05,
            reverse_force_node_order=False,
            columns_to_delete_zero_based=(20, 41),
            comparison_lines=(
                ComparisonLineSpec(
                    row_index_zero_based=0,
                    label="case_1",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-240-1.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=7,
                    label="case_2_centerline",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-240-2.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=12,
                    label="case_3",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-240-3.pdf",),
                    reverse_model_x=False,
                ),
            ),
        ),
        "double_270": YoonHingeCase(
            case_id="double_270",
            title="Yoon double hinge, 270 deg",
            module_count=3,
            module_rows=13,
            module_columns=21,
            total_nodes=819,
            mass_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_MASS1_rho282.mtx",
            stiffness_matrix_path=structure_root / "Job_hinge_study_100_60_YoonModel-1_STIF1_rho282.mtx",
            hydrodynamic_path=hydro_root / "DM10_direction270_slender180_rho1025.nc",
            hinges=double_hinges,
            master_nodes_one_based=tuple(
                sorted([130, 136, 142, 401, 407, 413, 419, 678, 684, 690])
            ),
            reduction_method="serep",
            hydrostatic_divisor=1.05,
            reverse_force_node_order=False,
            columns_to_delete_zero_based=(20, 41),
            comparison_lines=(
                ComparisonLineSpec(
                    row_index_zero_based=0,
                    label="case_1",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-270-1.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=7,
                    label="case_2_centerline",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-270-2.pdf",),
                    reverse_model_x=False,
                ),
                ComparisonLineSpec(
                    row_index_zero_based=12,
                    label="case_3",
                    legacy_figure_paths=(ref_figures / "Yoon-2-hige180-270-3.pdf",),
                    reverse_model_x=False,
                ),
            ),
        ),
    }
