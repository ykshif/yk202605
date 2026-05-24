from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    HydrodynamicExtrapolationConfig,
    build_extended_omega_grid,
    extrapolate_hydrodynamic_data,
    extrapolate_frequency_series,
    frequency_grid_diagnostics,
    hydrodynamic_array_diagnostics,
    max_abs_difference_inside_original_range,
)


class HydrodynamicExtrapolationTests(unittest.TestCase):
    def test_extended_omega_grid_embeds_original_values_exactly(self) -> None:
        omega = np.array([0.2, 0.4, 0.8])
        config = HydrodynamicExtrapolationConfig(
            low_frequency_min=0.05,
            high_frequency_max=1.4,
            low_frequency_count=2,
            high_frequency_count=3,
        )

        extended, original_slice = build_extended_omega_grid(omega, config)

        np.testing.assert_allclose(extended[original_slice], omega, rtol=0.0, atol=0.0)
        self.assertEqual(extended.size, omega.size + 5)
        self.assertTrue(np.all(np.diff(extended) > 0.0))

    def test_extrapolation_preserves_original_hydrodynamic_arrays(self) -> None:
        omega = np.array([0.2, 0.5, 1.0])
        added_mass = np.stack([np.eye(2) * (2.0 + value) for value in omega])
        damping = np.stack([np.eye(2) * value for value in [0.1, 0.4, 0.2]])
        force = np.array([[1.0 + 2.0j, 0.5], [2.0 + 1.0j, 0.25], [1.0 - 1.0j, 0.1]])
        config = HydrodynamicExtrapolationConfig(
            low_frequency_min=0.05,
            high_frequency_max=2.0,
            low_frequency_count=2,
            high_frequency_count=4,
        )

        extrapolated = extrapolate_hydrodynamic_data(
            omega,
            added_mass,
            damping,
            wave_force=force,
            config=config,
        )
        report = extrapolated.invariance_report(added_mass, damping, force)

        self.assertEqual(report["added_mass"], 0.0)
        self.assertEqual(report["radiation_damping"], 0.0)
        self.assertEqual(report["wave_force"], 0.0)
        self.assertLess(np.linalg.norm(extrapolated.radiation_damping[-1]), np.linalg.norm(damping[-1]))

    def test_max_abs_difference_inside_original_range_detects_changes(self) -> None:
        original = np.array([[1.0], [2.0]])
        extended = np.array([[0.0], [1.0], [2.25], [3.0]])

        delta = max_abs_difference_inside_original_range(original, extended, slice(1, 3))

        self.assertEqual(delta, 0.25)

    def test_frequency_series_extrapolates_force_like_shapes(self) -> None:
        omega = np.array([0.3, 0.6])
        force = np.ones((2, 2, 3), dtype=np.complex128)
        config = HydrodynamicExtrapolationConfig(
            low_frequency_min=0.1,
            high_frequency_max=1.2,
            low_frequency_count=1,
            high_frequency_count=2,
        )

        extended_omega, extended_force, original_slice = extrapolate_frequency_series(
            omega,
            force,
            config=config,
            series_kind="force",
        )

        self.assertEqual(extended_force.shape, (5, 2, 3))
        np.testing.assert_allclose(extended_omega[original_slice], omega)
        np.testing.assert_allclose(extended_force[original_slice], force)

    def test_diagnostics_report_grid_and_array_properties(self) -> None:
        omega = np.array([0.1, 0.2, 0.4])
        added_mass = np.stack([np.eye(2) for _ in omega])
        damping = np.stack([np.eye(2) * value for value in omega])

        grid = frequency_grid_diagnostics(omega, reference_omega=0.1)
        arrays = hydrodynamic_array_diagnostics(omega, added_mass, damping)

        self.assertTrue(grid["is_strictly_increasing"])
        self.assertFalse(grid["is_uniform"])
        self.assertTrue(arrays["added_mass_finite"])
        self.assertTrue(arrays["radiation_damping_finite"])


if __name__ == "__main__":
    unittest.main()
