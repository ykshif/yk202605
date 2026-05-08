"""Wind load utilities migrated from the legacy wind-load workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offshore_energy_sim.environment.spectra import (
    amplitude_from_spectrum,
    api_wind_spectrum,
    wind_speed_power_law,
)


@dataclass(frozen=True)
class WindGrid:
    """Grid and physical parameters for distributed wind loading."""

    total_rows: int
    total_cols: int
    area_m2: float = 2.0
    air_density: float = 1.225
    dofs_per_node: int = 6


def load_wind_coefficient_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column wind coefficient curve."""

    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def extend_coefficients_to_grid(
    coefficient_values: np.ndarray,
    total_rows: int,
    total_cols: int,
    reverse: bool = False,
) -> np.ndarray:
    """Extend 1D coefficient samples to a 2D wind-load coefficient grid."""

    extended = np.concatenate(
        [coefficient_values, [coefficient_values[-1]] * (total_cols - len(coefficient_values))]
    )
    if reverse:
        extended = extended[::-1]
    return np.tile(extended, (total_rows, 1))


def wind_amplitude_at_frequency(
    reference_speed: float,
    height: float,
    target_frequency: float,
    frequencies: np.ndarray | None = None,
    delta_frequency: float = 0.01,
    alpha: float = 0.125,
) -> float:
    """Return wind fluctuation amplitude at the nearest sampled frequency."""

    if frequencies is None:
        frequencies = np.arange(0.01, 2.0, 0.01)
    spectrum = api_wind_spectrum(reference_speed, height, frequencies, alpha=alpha)
    amplitudes = amplitude_from_spectrum(spectrum, delta_frequency)
    index = np.abs(frequencies - target_frequency).argmin()
    return float(amplitudes[index])


def distributed_wind_force(
    coefficient_grid: np.ndarray,
    reference_speed: float,
    height: float,
    target_frequency: float,
    grid: WindGrid,
    dof_index_zero_based: int = 0,
    alpha: float = 0.125,
) -> np.ndarray:
    """Build full distributed wind force vector for one structural DOF."""

    amplitude = wind_amplitude_at_frequency(
        reference_speed,
        height,
        target_frequency,
        alpha=alpha,
    )
    average_speed = wind_speed_power_law(reference_speed, height, alpha=alpha)
    one_dof_force = (
        2
        * coefficient_grid
        * average_speed
        * amplitude
        * grid.area_m2
        * grid.air_density
    ).reshape(grid.total_rows * grid.total_cols)

    force = np.zeros((1, grid.total_rows * grid.total_cols * grid.dofs_per_node), dtype=complex)
    force[0, dof_index_zero_based :: grid.dofs_per_node] = one_dof_force
    return force


def distributed_wind_damping(
    coefficient_grid: np.ndarray,
    reference_speed: float,
    height: float,
    grid: WindGrid,
    dof_index_zero_based: int = 0,
    alpha: float = 0.125,
) -> np.ndarray:
    """Build diagonal wind damping matrix for one structural DOF."""

    average_speed = wind_speed_power_law(reference_speed, height, alpha=alpha)
    damping_values = (
        2 * coefficient_grid * average_speed * grid.area_m2 * grid.air_density
    ).reshape(grid.total_rows * grid.total_cols)
    diagonal = np.zeros(grid.total_rows * grid.total_cols * grid.dofs_per_node)
    diagonal[dof_index_zero_based :: grid.dofs_per_node] = damping_values
    return np.diag(diagonal)


def split_submodule_coefficients(
    coefficient_grid: np.ndarray,
    num_submodules: int = 10,
    split_cols: int = 16,
) -> list[np.ndarray]:
    """Split wind coefficients into submodule blocks preserving legacy logic."""

    boundary_stride = split_cols - 1
    matrix = coefficient_grid.copy()
    matrix[:, boundary_stride::boundary_stride] /= 2
    boundary_columns = matrix[:, boundary_stride::boundary_stride][:, :-1]
    matrix_new = matrix.copy()

    for index in range(boundary_columns.shape[1]):
        insert_position = boundary_stride + boundary_stride * index + index
        matrix_new = np.insert(matrix_new, insert_position, boundary_columns[:, index], axis=1)

    return list(np.array_split(matrix_new, num_submodules, axis=1))
