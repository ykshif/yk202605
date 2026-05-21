"""Hydrodynamic data generation, loading, and access."""

from offshore_energy_sim.hydrodynamics.capytaine_array import (
    ArrayHydrodynamicsConfig,
    ArrayHydrodynamicsResult,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    build_array_body,
    build_hydrodynamic_problems,
    degrees_to_radians,
    module_structural_node_mappings,
    omega_from_wavelength,
    omega_values_from_range,
    omega_values_from_wavelengths,
    parse_float_sequence,
    preview_layout,
    run_array_hydrodynamics,
    summarize_rao_for_ui,
)
from offshore_energy_sim.hydrodynamics.netcdf import (
    HydrodynamicDatasetSummary,
    open_hydrodynamic_dataset,
    summarize_hydrodynamic_dataset,
)
from offshore_energy_sim.hydrodynamics.frequency import (
    HydrodynamicTerms,
    prepare_hydrodynamic_terms,
    reverse_hydrodynamic_node_order_force,
    reverse_hydrodynamic_node_order_matrix,
)

__all__ = [
    "ArrayHydrodynamicsConfig",
    "ArrayHydrodynamicsResult",
    "ArrayLayoutSpec",
    "HydrodynamicDatasetSummary",
    "HydrodynamicTerms",
    "RectangularModuleSpec",
    "StructuralGridSpec",
    "build_array_body",
    "build_hydrodynamic_problems",
    "degrees_to_radians",
    "module_structural_node_mappings",
    "omega_from_wavelength",
    "omega_values_from_range",
    "omega_values_from_wavelengths",
    "open_hydrodynamic_dataset",
    "parse_float_sequence",
    "prepare_hydrodynamic_terms",
    "preview_layout",
    "reverse_hydrodynamic_node_order_force",
    "reverse_hydrodynamic_node_order_matrix",
    "run_array_hydrodynamics",
    "summarize_rao_for_ui",
    "summarize_hydrodynamic_dataset",
]
