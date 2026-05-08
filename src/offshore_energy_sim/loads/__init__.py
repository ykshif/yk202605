"""Wave, wind, and combined load models."""

from offshore_energy_sim.loads.vector_mapping import extend_force_vector_to_nodes
from offshore_energy_sim.loads.wind import (
    WindGrid,
    distributed_wind_damping,
    distributed_wind_force,
    extend_coefficients_to_grid,
    load_wind_coefficient_curve,
    split_submodule_coefficients,
    wind_amplitude_at_frequency,
)

__all__ = [
    "WindGrid",
    "distributed_wind_damping",
    "distributed_wind_force",
    "extend_coefficients_to_grid",
    "extend_force_vector_to_nodes",
    "load_wind_coefficient_curve",
    "split_submodule_coefficients",
    "wind_amplitude_at_frequency",
]
