"""Validation workflows for published benchmark cases."""

from offshore_energy_sim.validation.complex_hinge_10x10 import (
    ComplexHingeCase,
    ComplexHingeResult,
    build_complex_hinge_10x10_case,
    missing_input_paths as missing_complex_hinge_input_paths,
    plot_complex_hinge_result,
    solve_complex_hinge_case,
)
from offshore_energy_sim.validation.modular_hinge_grid import (
    build_modular_hinge_grid_case,
    default_hydrodynamic_output_path,
    default_module_size_m,
    legacy_hydrodynamic_path_for_grid,
)
from offshore_energy_sim.validation.yoon_hinge import (
    ComparisonLineSpec,
    YoonHingeCase,
    YoonHingeResult,
    build_yoon_hinge_cases,
    missing_input_paths,
    plot_yoon_hinge_case,
    solve_yoon_hinge_case,
)

__all__ = [
    "ComparisonLineSpec",
    "ComplexHingeCase",
    "ComplexHingeResult",
    "YoonHingeCase",
    "YoonHingeResult",
    "build_complex_hinge_10x10_case",
    "build_modular_hinge_grid_case",
    "build_yoon_hinge_cases",
    "default_hydrodynamic_output_path",
    "default_module_size_m",
    "legacy_hydrodynamic_path_for_grid",
    "missing_complex_hinge_input_paths",
    "missing_input_paths",
    "plot_complex_hinge_result",
    "plot_yoon_hinge_case",
    "solve_complex_hinge_case",
    "solve_yoon_hinge_case",
]
