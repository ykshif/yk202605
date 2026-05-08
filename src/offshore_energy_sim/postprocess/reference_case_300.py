"""Reference-case utilities for the 300 m x 60 m floating body.

This module centralizes the read-only validation and plotting logic that was
previously duplicated in scripts. It intentionally does not run or modify the
hydroelastic solver. Numerical algorithms and baseline data are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offshore_energy_sim.core.cases import (
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)
from offshore_energy_sim.environment.waves import RegularWave
from offshore_energy_sim.geometry.floating_body import RectangularFloatingBody
from offshore_energy_sim.hydrodynamics.netcdf import summarize_hydrodynamic_dataset
from offshore_energy_sim.postprocess.metrics import rmse
from offshore_energy_sim.postprocess.plots import plot_heave_rao_comparison
from offshore_energy_sim.response.retained_dofs import retained_node_dof_series
from offshore_energy_sim.structure.matrix_io import scan_abaqus_matrix_file
from offshore_energy_sim.utils.hashing import sha256_file


CASE_TITLE = "300 m x 60 m floating body, wavelength 300 m"
TOLERANCE = 1e-12

GEOMETRY = RectangularFloatingBody(
    length_m=300.0,
    width_m=60.0,
    thickness_m=2.0,
    total_nodes=793,
    retained_dofs_per_node=5,
    mesh_label="5m_x_5m",
)
WAVE = RegularWave(wavelength_m=300.0, direction_deg=0.0)


@dataclass(frozen=True)
class ReferenceCase300Paths:
    """Filesystem locations for the 300 m x 60 m baseline case."""

    repo_root: Path
    response_file: Path
    hydrodynamic_file: Path
    structural_mass_file: Path
    structural_stiffness_file: Path
    experiment_file: Path
    fu_sim_file: Path
    figure_dir: Path

    @property
    def png_path(self) -> Path:
        return self.figure_dir / "reference_case_300_heave_vs_experiment.png"

    @property
    def pdf_path(self) -> Path:
        return self.figure_dir / "reference_case_300_heave_vs_experiment.pdf"


@dataclass(frozen=True)
class ReferenceCase300Verification:
    """Verification result for the baseline response and comparison data."""

    paths: ReferenceCase300Paths
    metrics: dict[str, object]
    hash_failures: list[str]
    metric_failures: list[str]

    @property
    def failures(self) -> list[str]:
        return self.hash_failures + self.metric_failures

    @property
    def passed(self) -> bool:
        return not self.failures


def default_paths(repo_root: Path | None = None) -> ReferenceCase300Paths:
    """Build default paths for the documented local baseline case."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root).resolve()
    dm_fem_root = Path(r"E:\phd\Code\DM-FEM2D")
    comparison_dir = Path(r"E:\phd\Code\DM-FEM2D\data\Experiment_300_60")
    return ReferenceCase300Paths(
        repo_root=repo_root,
        response_file=repo_root / "displacement_55mesh_300.npy",
        hydrodynamic_file=dm_fem_root / "HydrodynamicData" / "Yoga" / "DM10_300_direction0.nc",
        structural_mass_file=dm_fem_root / "StructureData" / "JobMesh5_5_MASS1.mtx",
        structural_stiffness_file=dm_fem_root / "StructureData" / "JobMesh5_5_STIF1.mtx",
        experiment_file=comparison_dir / "exp_300.txt",
        fu_sim_file=comparison_dir / "fu_sim300.txt",
        figure_dir=repo_root / "figures",
    )


def build_rodm_frequency_case(
    paths: ReferenceCase300Paths | None = None,
    reverse_hydrodynamic_node_order: bool = False,
) -> RodmFrequencyCase:
    """Build the standardized RODM case model for this baseline."""

    paths = paths or default_paths()
    return RodmFrequencyCase(
        case_id="reference_case_300",
        total_nodes=793,
        full_dofs_per_node=6,
        retained_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        master_node_rule=MasterNodeRule(first_node=424, node_interval=6, count=10),
        hydrodynamic_dataset=paths.hydrodynamic_file,
        structural_matrices=StructuralMatrixPaths(
            mass=paths.structural_mass_file,
            stiffness=paths.structural_stiffness_file,
        ),
        hydrodynamic_nodes=10,
        hydrodynamic_dof_to_remove_zero_based=5,
        mass_blend_beta=0.0,
        use_hydrostatic=True,
        frequency_index=0,
        reverse_hydrodynamic_node_order=reverse_hydrodynamic_node_order,
    )


def expected_hashes(paths: ReferenceCase300Paths) -> dict[Path, str]:
    """Expected file digests for the current baseline artifacts."""

    return {
        paths.response_file: "1BE5B04A036857AD71E480C772D2CDA0FC1C0850D6CD9AA845B67F3EBEFD8DA5",
        paths.hydrodynamic_file: "D2414083E634B958139C5A4203BFD2C7AFA1782D34D4A80F0F12E669BD8EEEC9",
        paths.structural_mass_file: "FDB09EB5149417A0EE3BAB01827F128EF6F1D2A82A0A56709A65422E8A45009B",
        paths.structural_stiffness_file: "4D7B48381323F35210A38469A4F8BC81533FFC57473682BF2108E2A69C5566AA",
        paths.experiment_file: "59E3ED95F7069A798638332238BD780C13F435F6F9DFC17391AAE32D965CACC2",
        paths.fu_sim_file: "8F386D41095B9992949C9EBB939E9F1E6262FC460EC9522A2E241B1149505E07",
    }


EXPECTED_METRICS = {
    "response_shape": (3965, 1),
    "heave_len": 60,
    "heave_abs_min": 0.8166492461156475,
    "heave_abs_max": 1.2525429563334871,
    "heave_abs_mean": 0.896981533364751,
    "heave_abs_l2": 7.000919304253492,
    "rmse_vs_exp300": 0.06367482251124734,
    "rmse_vs_fu_sim300": 0.04488934895346538,
}


def load_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column ``x, y`` comparison curve."""

    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def load_response(paths: ReferenceCase300Paths | None = None) -> np.ndarray:
    """Load the full retained-DOF response vector for the baseline case."""

    paths = paths or default_paths()
    return np.load(paths.response_file)


def extract_centerline_heave(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract centerline heave RAO from the retained-DOF response vector.

    The response has shape ``(793 nodes * 5 retained DOFs, 1)``. Each retained
    node block stores 5 structural DOFs; local index 2 is vertical heave.
    """

    heave_complex = retained_node_dof_series(
        response,
        start_node_one_based=367,
        stop_node_one_based=427,
        retained_dofs_per_node=5,
        dof_index_zero_based=2,
        column=0,
    )
    heave_rao = np.abs(heave_complex)
    x_over_l = np.linspace(0.0, 1.0, heave_rao.size)
    return x_over_l, heave_rao


def load_present_heave(
    paths: ReferenceCase300Paths | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and extract the present RODM centerline heave RAO."""

    return extract_centerline_heave(load_response(paths))


def compute_metrics(
    paths: ReferenceCase300Paths | None = None,
) -> dict[str, object]:
    """Compute reproducibility metrics for the 300 m x 60 m baseline."""

    paths = paths or default_paths()
    response = load_response(paths)
    x_present, heave = extract_centerline_heave(response)

    exp_x, exp_y = load_xy(paths.experiment_file)
    fu_x, fu_y = load_xy(paths.fu_sim_file)

    present_at_exp = np.interp(exp_x, x_present, heave)
    present_at_fu = np.interp(fu_x, x_present, heave)

    return {
        "response_shape": response.shape,
        "response_dtype": str(response.dtype),
        "heave_len": heave.size,
        "heave_abs_min": float(np.min(heave)),
        "heave_abs_max": float(np.max(heave)),
        "heave_abs_mean": float(np.mean(heave)),
        "heave_abs_l2": float(np.linalg.norm(heave)),
        "heave_first5": [float(value) for value in heave[:5]],
        "heave_last5": [float(value) for value in heave[-5:]],
        "exp_points": exp_x.size,
        "fu_points": fu_x.size,
        "rmse_vs_exp300": rmse(exp_y, present_at_exp),
        "rmse_vs_fu_sim300": rmse(fu_y, present_at_fu),
    }


def summarize_input_files(paths: ReferenceCase300Paths | None = None) -> dict[str, object]:
    """Summarize read-only input files used by the documented baseline case."""

    paths = paths or default_paths()
    return {
        "hydrodynamic": summarize_hydrodynamic_dataset(
            paths.hydrodynamic_file,
            load_metadata=True,
        ),
        "structural_mass": scan_abaqus_matrix_file(paths.structural_mass_file),
        "structural_stiffness": scan_abaqus_matrix_file(paths.structural_stiffness_file),
    }


def format_input_summary_report(paths: ReferenceCase300Paths | None = None) -> str:
    """Format hydrodynamic and structural input metadata for console output."""

    paths = paths or default_paths()
    summaries = summarize_input_files(paths)
    hydro = summaries["hydrodynamic"]
    mass = summaries["structural_mass"]
    stiffness = summaries["structural_stiffness"]

    lines = [
        f"Reference case inputs: {CASE_TITLE}",
        "Mode: read-only input validation",
        "",
        "Hydrodynamic NetCDF:",
        f"  exists: {hydro.exists}",
        f"  path: {hydro.path}",
        f"  sha256: {hydro.sha256}",
        f"  xarray_available: {hydro.xarray_available}",
        f"  capytaine_available: {hydro.capytaine_available}",
    ]
    if hydro.dims is not None:
        lines.append(f"  dims: {hydro.dims}")
        lines.append(f"  data_variables: {hydro.data_variables}")
    if hydro.note:
        lines.append(f"  note: {hydro.note}")

    for label, summary in (("Structural mass matrix", mass), ("Structural stiffness matrix", stiffness)):
        lines.extend(
            [
                "",
                f"{label}:",
                f"  exists: {summary.exists}",
                f"  path: {summary.path}",
                f"  sha256: {summary.sha256}",
                f"  dofs_per_node: {summary.dofs_per_node}",
                f"  max_node_id: {summary.max_node_id}",
                f"  shape: {summary.shape}",
                f"  stored_entries: {summary.stored_entries}",
                f"  symmetric_entries_estimate: {summary.symmetric_entries_estimate}",
            ]
        )

    hash_failures = verify_hashes(paths)
    if hash_failures:
        lines.extend(["", "Input hash verification failed:"])
        lines.extend(f"  - {failure}" for failure in hash_failures)
    else:
        lines.extend(["", "Input hash verification passed."])

    return "\n".join(lines)


def verify_hashes(paths: ReferenceCase300Paths) -> list[str]:
    """Check that baseline files still match their documented digests."""

    failures = []
    for path, expected in expected_hashes(paths).items():
        if not path.exists():
            failures.append(f"Missing file: {path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    return failures


def verify_metrics(
    metrics: dict[str, object],
    tolerance: float = TOLERANCE,
) -> list[str]:
    """Compare computed baseline metrics with documented values."""

    failures = []
    for name, expected in EXPECTED_METRICS.items():
        actual = metrics[name]
        if isinstance(expected, tuple):
            if tuple(actual) != expected:
                failures.append(f"{name}: expected {expected}, got {actual}")
            continue

        if abs(float(actual) - expected) > tolerance:
            failures.append(f"{name}: expected {expected}, got {actual}")
    return failures


def verify_reference_case_300(
    paths: ReferenceCase300Paths | None = None,
    tolerance: float = TOLERANCE,
) -> ReferenceCase300Verification:
    """Run read-only baseline verification."""

    paths = paths or default_paths()
    metrics = compute_metrics(paths)
    return ReferenceCase300Verification(
        paths=paths,
        metrics=metrics,
        hash_failures=verify_hashes(paths),
        metric_failures=verify_metrics(metrics, tolerance),
    )


def format_verification_report(result: ReferenceCase300Verification) -> str:
    """Format a console report matching the legacy script behavior."""

    lines = [
        f"Reference case: {CASE_TITLE}",
        "Mode: read-only baseline verification",
        "",
        "Files:",
    ]
    for path in expected_hashes(result.paths):
        status = "exists" if path.exists() else "missing"
        lines.append(f"  {status}: {path}")

    lines.extend(["", "Metrics:"])
    for name, value in result.metrics.items():
        lines.append(f"  {name}: {value}")

    if result.failures:
        lines.extend(["", "Verification failed:"])
        lines.extend(f"  - {failure}" for failure in result.failures)
    else:
        lines.extend(["", "Verification passed: baseline files and numerical metrics match."])

    return "\n".join(lines)


def plot_reference_case_300(
    paths: ReferenceCase300Paths | None = None,
    include_fu: bool = True,
) -> tuple[Path, Path]:
    """Plot present RODM heave RAO against experiment and Fu et al. data."""

    paths = paths or default_paths()
    x_present, heave_present = load_present_heave(paths)
    x_exp, heave_exp = load_xy(paths.experiment_file)
    x_fu = None
    heave_fu = None
    if include_fu and paths.fu_sim_file.exists():
        x_fu, heave_fu = load_xy(paths.fu_sim_file)

    return plot_heave_rao_comparison(
        x_present,
        heave_present,
        x_exp,
        heave_exp,
        paths.png_path,
        paths.pdf_path,
        x_reference=x_fu,
        heave_reference=heave_fu,
        reference_label="Fu et al. (2007)",
        title="300 m x 60 m Floating Body, Wavelength 300 m",
    )
