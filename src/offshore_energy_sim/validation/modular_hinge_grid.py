"""Generic square modular hinge hydroelastic cases.

The original validated implementation lives in ``complex_hinge_10x10.py`` and
is notebook-compatible for the 10 x 10, 30 m module case.  This module exposes
the same numerical path for arbitrary square grids so paper scripts can
inventory and, when the required hydrodynamic and structural inputs exist,
evaluate 5 x 5, 10 x 10, and 15 x 15 discretizations with one consistent API.
"""

from __future__ import annotations

from pathlib import Path

from offshore_energy_sim.structure import (
    ModuleGridSpec,
    generate_grid_hinge_specs,
    generate_master_nodes_one_based,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (
    ComplexHingeCase,
    complex_hinge_data_root,
)


def default_module_size_m(modules_per_side: int, *, total_size_m: float = 300.0) -> float:
    """Return the square module side length for a fixed total platform size."""

    if int(modules_per_side) <= 0:
        raise ValueError("modules_per_side must be positive")
    return float(total_size_m) / int(modules_per_side)


def default_hydrodynamic_output_path(
    modules_per_side: int,
    *,
    data_root: str | Path | None = None,
    total_size_m: float = 300.0,
    omega: float = 0.5851,
    direction_deg: float = 0.0,
) -> Path:
    """Return the standardized generated hydrodynamic path for one grid."""

    root = complex_hinge_data_root() if data_root is None else Path(data_root)
    module_size = default_module_size_m(modules_per_side, total_size_m=total_size_m)
    omega_label = f"{float(omega):.4f}".replace(".", "p")
    direction_label = f"{float(direction_deg):g}".replace(".", "p")
    module_label = f"{module_size:.6g}".replace(".", "p")
    return (
        root
        / "HydrodynamicData"
        / "ModularGrid"
        / (
            f"DM{modules_per_side}x{modules_per_side}"
            f"_L{module_label}_omega{omega_label}_dir{direction_label}.nc"
        )
    )


def legacy_hydrodynamic_path_for_grid(
    modules_per_side: int,
    *,
    data_root: str | Path | None = None,
) -> Path | None:
    """Return a legacy hydrodynamic path when one is known."""

    root = complex_hinge_data_root() if data_root is None else Path(data_root)
    if int(modules_per_side) == 10:
        return root / "HydrodynamicData" / "Yoon_hinge" / "DM10_10_direction0_wl180.nc"
    return None


def build_modular_hinge_grid_case(
    modules_per_side: int,
    data_root: str | Path | None = None,
    *,
    total_size_m: float = 300.0,
    nodes_per_module_side: int = 7,
    center_node_one_based: int | None = None,
    k_hinge: float = 1.0e10,
    released_dof_stiffness: float = 10.0,
    mass_matrix_path: str | Path | None = None,
    stiffness_matrix_path: str | Path | None = None,
    hydrodynamic_path: str | Path | None = None,
    hydrostatic_divisor: float = 1.05,
    frequency_index: int = 0,
) -> ComplexHingeCase:
    """Build a square-grid hinge case using the validated 10 x 10 solver path.

    Parameters
    ----------
    modules_per_side:
        Number of modules along x and y.  For the paper comparison this is
        normally 5, 10, or 15.
    total_size_m:
        Overall side length of the platform.  The module side length is
        ``total_size_m / modules_per_side``.

    Notes
    -----
    The default structural matrix paths intentionally point to the existing
    30 m x 30 m module matrix.  That default is valid for reproducing the
    validated 10 x 10 case.  For 5 x 5 and 15 x 15 publication-quality
    response comparisons, pass structure matrices generated for the matching
    60 m or 20 m module model.
    """

    n = int(modules_per_side)
    if n <= 0:
        raise ValueError("modules_per_side must be positive")
    data_root = complex_hinge_data_root() if data_root is None else Path(data_root)
    module_size = default_module_size_m(n, total_size_m=total_size_m)
    if center_node_one_based is None:
        center_node_one_based = (int(nodes_per_module_side) ** 2 + 1) // 2
    grid = ModuleGridSpec(
        modules_per_side=n,
        nodes_per_module_side=nodes_per_module_side,
        module_size=module_size,
        center_node_one_based=int(center_node_one_based),
    )
    structure_root = data_root / "StructureData" / "Hinge_complex_paper4"
    legacy_hydro = legacy_hydrodynamic_path_for_grid(n, data_root=data_root)
    default_hydro = legacy_hydro or default_hydrodynamic_output_path(
        n,
        data_root=data_root,
        total_size_m=total_size_m,
    )
    hinges = generate_grid_hinge_specs(
        grid,
        k_hinge=k_hinge,
        released_dof_stiffness=released_dof_stiffness,
    )
    return ComplexHingeCase(
        case_id=f"modular_hinge_{n}x{n}_total{float(total_size_m):g}m_wl180_dir0",
        title=(
            f"{n}x{n} modular hinge hydroelastic case, "
            f"{module_size:g} m modules, total side {float(total_size_m):g} m"
        ),
        grid=grid,
        mass_matrix_path=(
            Path(mass_matrix_path)
            if mass_matrix_path is not None
            else structure_root / "Job3030hinge-1_MASS1.mtx"
        ),
        stiffness_matrix_path=(
            Path(stiffness_matrix_path)
            if stiffness_matrix_path is not None
            else structure_root / "Job3030hinge-1_STIF1.mtx"
        ),
        hydrodynamic_path=Path(hydrodynamic_path) if hydrodynamic_path is not None else default_hydro,
        hinges=hinges,
        master_nodes_one_based=generate_master_nodes_one_based(grid),
        hydrostatic_divisor=hydrostatic_divisor,
        frequency_index=frequency_index,
        hydrodynamic_nodes=n * n,
        source_programs=(
            "offshore_energy_sim.validation.modular_hinge_grid",
            "offshore_energy_sim.validation.complex_hinge_10x10",
        ),
    )
