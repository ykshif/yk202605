"""Validate environment, load, response, strength, and PV helper kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.environment import (  # noqa: E402
    amplitude_from_spectrum,
    api_wind_spectrum,
    jonswap_spectrum,
    turbulence_intensity_api,
    wind_speed_power_law,
)
from offshore_energy_sim.loads import (  # noqa: E402
    WindGrid,
    distributed_wind_damping,
    distributed_wind_force,
    extend_coefficients_to_grid,
    split_submodule_coefficients,
)
from offshore_energy_sim.power import (  # noqa: E402
    cosine_incidence_factor,
    dc_power_from_irradiance,
    power_with_tilt_loss,
    relative_power_loss,
)
from offshore_energy_sim.response import response_spectrum_from_amplitude, rms_from_spectrum  # noqa: E402
from offshore_energy_sim.strength import (  # noqa: E402
    compute_module_forces,
    extract_module_displacements,
    generate_1d_module_nodes,
    map_module_forces_to_global_nodes,
    middle_interface_moment_per_width,
)


def assert_allclose(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def validate_jonswap() -> None:
    omega = np.array([0.5, 1.0, 1.5])
    actual = jonswap_spectrum(1.25, 8.29, omega)
    peak = 2 * np.pi / 8.29
    sigma = np.where(omega <= peak, 0.07, 0.09)
    gamma = 3.3
    alpha = 0.0624 / (0.230 + 0.0336 * gamma - (0.185 / (1.9 + gamma)))
    beta = np.exp(-((omega - peak) ** 2) / (2 * (sigma**2) * (peak**2)))
    expected = alpha * 1.25**2 * peak**4 * omega ** (-5) * gamma**beta
    expected *= np.exp(-1.25 * (peak / omega) ** 4) * 2 * np.pi
    assert_allclose("jonswap_spectrum", actual, expected)


def validate_wind_spectrum_and_grid_loads() -> None:
    frequencies = np.array([0.01, 0.02, 0.03])
    spectrum = api_wind_spectrum(14.3, 2.0, frequencies)
    adjusted_speed = wind_speed_power_law(14.3, 2.0)
    turbulence = turbulence_intensity_api(2.0)
    peak_frequency = 0.025 * adjusted_speed / 2.0
    expected_spectrum = (
        adjusted_speed**2
        * turbulence**2
        / peak_frequency
        * (1 + 1.5 * (frequencies / peak_frequency)) ** (-5 / 3)
    )
    assert_allclose("api_wind_spectrum", spectrum, expected_spectrum)
    assert_allclose("amplitude_from_spectrum", amplitude_from_spectrum(spectrum, 0.01), np.sqrt(2 * spectrum * 0.01))

    coeff_grid = extend_coefficients_to_grid(np.array([1.0, 2.0]), total_rows=2, total_cols=4)
    assert_allclose(
        "extend_coefficients_to_grid",
        coeff_grid,
        np.array([[1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0]]),
    )

    grid = WindGrid(total_rows=2, total_cols=2, area_m2=2.0, air_density=1.225, dofs_per_node=3)
    small_coeff = np.ones((2, 2))
    force = distributed_wind_force(small_coeff, 14.3, 2.0, 0.02, grid, dof_index_zero_based=1)
    damping = distributed_wind_damping(small_coeff, 14.3, 2.0, grid, dof_index_zero_based=1)
    if force.shape != (1, 12):
        raise AssertionError(f"unexpected wind force shape: {force.shape}")
    if damping.shape != (12, 12):
        raise AssertionError(f"unexpected wind damping shape: {damping.shape}")
    if not np.all(force[0, 0::3] == 0) or not np.all(force[0, 2::3] == 0):
        raise AssertionError("wind force should only occupy the selected DOF")

    modules = split_submodule_coefficients(np.ones((2, 5)), num_submodules=2, split_cols=3)
    if len(modules) != 2:
        raise AssertionError("split_submodule_coefficients should return two blocks")


def validate_strength_helpers() -> None:
    modules = generate_1d_module_nodes(total_nodes=12, rows=3, module_nodes=3, module_number=2)
    expected_modules = [[1, 2, 3, 5, 6, 7, 9, 10, 11], [3, 4, 5, 7, 8, 9, 11, 12, 13]]
    if modules != expected_modules:
        raise AssertionError(f"module generation mismatch: {modules}")

    displacement = np.arange(24, dtype=float).reshape(24, 1)
    module_displacements = extract_module_displacements(
        displacement,
        [modules[0]],
        total_nodes=12,
        dofs_per_node=2,
    )
    stiffness = np.eye(len(modules[0]) * 2) * 3
    module_forces = compute_module_forces(stiffness, module_displacements)
    global_forces = map_module_forces_to_global_nodes(
        module_forces,
        [modules[0]],
        total_nodes=12,
        dofs_per_node=2,
    )
    assert_allclose("module_force_mapping_node1", global_forces[0], np.array([0.0, 3.0]))

    interface = middle_interface_moment_per_width(
        global_forces,
        total_nodes=12,
        rows=3,
        module_nodes=3,
        element_width=2.0,
        dofs_per_node=2,
        moment_dof_zero_based=1,
    )
    if interface.size == 0:
        raise AssertionError("interface force extraction should not be empty")


def validate_power_and_response_helpers() -> None:
    reference = dc_power_from_irradiance(np.array([1000.0, 800.0]), 2.0, 0.2)
    actual = power_with_tilt_loss(np.array([1000.0, 800.0]), 2.0, 0.2, np.array([0.0, np.pi / 3]))
    assert_allclose("cosine_incidence_factor", cosine_incidence_factor(np.array([0.0, np.pi])), np.array([1.0, 0.0]))
    assert_allclose("relative_power_loss", relative_power_loss(reference, actual), np.array([0.0, 0.5]))

    amplitudes = np.array([2.0, 3.0])
    spectrum = response_spectrum_from_amplitude(amplitudes, delta_frequency=0.5)
    assert_allclose("response_spectrum_from_amplitude", spectrum, np.array([8.0, 18.0]))
    assert_allclose("rms_from_spectrum", rms_from_spectrum(spectrum, 0.5), np.sqrt(13.0))


def main() -> int:
    validations = [
        validate_jonswap,
        validate_wind_spectrum_and_grid_loads,
        validate_strength_helpers,
        validate_power_and_response_helpers,
    ]
    for validation in validations:
        validation()
        print(f"passed: {validation.__name__}")

    print("Environment/load/power/strength validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
