from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    DiscreteStateSpaceRadiationModel,
    StateSpaceRadiationLinearSystem,
    StateSpaceRadiationModel,
    evaluate_era_markov_parameters,
    evaluate_state_space_radiation_kernel,
    fit_era_state_space_radiation,
    fit_common_pole_state_space_radiation,
    load_discrete_state_space_radiation_model,
    save_discrete_state_space_radiation_model,
    simulate_era_memory_force,
    simulate_state_space_memory_force,
    solve_state_space_radiation_linear_system,
    solve_state_space_radiation_linear_system_rk4,
    state_space_radiation_coefficients,
)
from offshore_energy_sim.time_domain import solve_linear_time_domain  # noqa: E402
import tempfile


class StateSpaceRadiationTests(unittest.TestCase):
    def test_common_pole_fit_recovers_exponential_kernel(self) -> None:
        time = np.linspace(0.0, 8.0, 161)
        poles = np.array([0.4, 1.7])
        residues = np.array(
            [
                [[2.0, 0.3], [0.3, 1.1]],
                [[-0.2, 0.1], [0.1, 0.7]],
            ]
        )
        reference = np.einsum("tp,pij->tij", np.exp(-np.outer(time, poles)), residues)

        model = fit_common_pole_state_space_radiation(
            time,
            reference,
            poles=poles,
            enforce_symmetric_residues=True,
        )
        fitted = evaluate_state_space_radiation_kernel(model, time)

        self.assertLess(model.fit_l2_relative_error, 1.0e-12)
        np.testing.assert_allclose(fitted, reference, atol=1.0e-12)

    def test_state_space_frequency_coefficients_match_analytic_scalar(self) -> None:
        model = StateSpaceRadiationModel(
            poles=np.array([2.0]),
            residues=np.array([[[6.0]]]),
            fit_l2_relative_error=0.0,
            fit_peak_relative_error=0.0,
        )
        omega = np.array([0.5, 1.0, 3.0])
        a_inf = np.array([[10.0]])

        added, damping = state_space_radiation_coefficients(
            model,
            omega,
            added_mass_infinite=a_inf,
        )

        expected_damping = 6.0 * 2.0 / (2.0**2 + omega**2)
        expected_added = 10.0 - 6.0 / (2.0**2 + omega**2)
        np.testing.assert_allclose(damping[:, 0, 0], expected_damping)
        np.testing.assert_allclose(added[:, 0, 0], expected_added)

    def test_state_space_memory_force_shapes_and_is_finite(self) -> None:
        time = np.linspace(0.0, 4.0, 81)
        velocity = np.column_stack([np.sin(time), np.cos(0.7 * time)])
        model = StateSpaceRadiationModel(
            poles=np.array([0.8, 2.2]),
            residues=np.array(
                [
                    [[1.0, 0.2], [0.2, 0.5]],
                    [[0.1, 0.0], [0.0, 0.3]],
                ]
            ),
            fit_l2_relative_error=0.0,
            fit_peak_relative_error=0.0,
        )

        force = simulate_state_space_memory_force(velocity, time, model)

        self.assertEqual(force.shape, velocity.shape)
        self.assertTrue(np.all(np.isfinite(force)))
        np.testing.assert_allclose(force[0], 0.0)

    def test_era_recovers_first_order_markov_sequence(self) -> None:
        time = np.linspace(0.0, 1.0, 101)
        dt = time[1] - time[0]
        pole = 0.94
        residue = np.array([[2.0]])
        kernel = np.zeros((time.size, 1, 1))
        for lag in range(1, time.size):
            kernel[lag] = residue * pole ** (lag - 1) / dt

        model = fit_era_state_space_radiation(
            time,
            kernel,
            order=1,
            block_rows=8,
            block_cols=8,
            stabilize=False,
        )
        markov = evaluate_era_markov_parameters(model, 20)
        reference = dt * kernel[1:21]

        self.assertLess(model.fit_l2_relative_error, 1.0e-10)
        np.testing.assert_allclose(markov, reference, atol=1.0e-10)

    def test_era_memory_force_shapes_and_is_finite(self) -> None:
        time = np.linspace(0.0, 2.0, 101)
        kernel = np.exp(-time)[:, np.newaxis, np.newaxis]
        model = fit_era_state_space_radiation(time, kernel, order=2, block_rows=8, block_cols=8)
        velocity = np.sin(time)[:, np.newaxis]

        force = simulate_era_memory_force(velocity, time, model)

        self.assertEqual(force.shape, velocity.shape)
        self.assertTrue(np.all(np.isfinite(force)))
        np.testing.assert_allclose(force[0], 0.0)

    def test_state_space_solver_matches_direct_convolution_for_discrete_kernel(self) -> None:
        time = np.linspace(0.0, 4.0, 161)
        dt = float(time[1] - time[0])
        pole = 0.96
        gain = 0.08
        zero_lag = np.array([[0.2]])
        kernel = np.zeros((time.size, 1, 1))
        kernel[0] = zero_lag
        for lag in range(1, time.size):
            kernel[lag, 0, 0] = gain * pole ** (lag - 1) / dt
        model = DiscreteStateSpaceRadiationModel(
            state_matrix=np.array([[pole]]),
            input_matrix=np.array([[1.0]]),
            output_matrix=np.array([[gain]]),
            time_step=dt,
            zero_lag_kernel=zero_lag,
            fit_l2_relative_error=0.0,
            spectral_radius=pole,
        )
        mass = np.array([[1.4]])
        damping = np.array([[0.03]])
        stiffness = np.array([[0.7]])
        force = np.sin(1.3 * time)[:, np.newaxis]

        direct = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            radiation_irf=kernel,
            radiation_convolution_rule="rectangular",
        )
        state = solve_state_space_radiation_linear_system(
            StateSpaceRadiationLinearSystem(
                mass=mass,
                damping=damping,
                stiffness=stiffness,
                force=force,
                time=time,
                radiation_model=model,
            ),
            radiation_convolution_rule="rectangular",
        )

        np.testing.assert_allclose(state.displacement, direct.displacement, atol=1.0e-12)
        np.testing.assert_allclose(state.velocity, direct.velocity, atol=1.0e-12)
        np.testing.assert_allclose(state.memory_force, direct.memory_force, atol=1.0e-12)

    def test_state_space_rk4_matches_direct_rk4_for_discrete_kernel(self) -> None:
        time = np.linspace(0.0, 4.0, 161)
        dt = float(time[1] - time[0])
        pole = 0.96
        gain = 0.08
        zero_lag = np.array([[0.2]])
        kernel = np.zeros((time.size, 1, 1))
        kernel[0] = zero_lag
        for lag in range(1, time.size):
            kernel[lag, 0, 0] = gain * pole ** (lag - 1) / dt
        model = DiscreteStateSpaceRadiationModel(
            state_matrix=np.array([[pole]]),
            input_matrix=np.array([[1.0]]),
            output_matrix=np.array([[gain]]),
            time_step=dt,
            zero_lag_kernel=zero_lag,
            fit_l2_relative_error=0.0,
            spectral_radius=pole,
        )
        mass = np.array([[1.4]])
        damping = np.array([[0.03]])
        stiffness = np.array([[0.7]])
        force = np.sin(1.3 * time)[:, np.newaxis]

        direct = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            radiation_irf=kernel,
            radiation_convolution_rule="rectangular",
            integrator="rk4",
        )
        state = solve_state_space_radiation_linear_system_rk4(
            StateSpaceRadiationLinearSystem(
                mass=mass,
                damping=damping,
                stiffness=stiffness,
                force=force,
                time=time,
                radiation_model=model,
            ),
            radiation_convolution_rule="rectangular",
        )

        np.testing.assert_allclose(state.displacement, direct.displacement, atol=1.0e-12)
        np.testing.assert_allclose(state.velocity, direct.velocity, atol=1.0e-12)
        np.testing.assert_allclose(state.memory_force, direct.memory_force, atol=1.0e-12)

    def test_discrete_state_space_model_save_load_roundtrip(self) -> None:
        model = DiscreteStateSpaceRadiationModel(
            state_matrix=np.array([[0.9, 0.1], [0.0, 0.8]]),
            input_matrix=np.array([[1.0], [0.5]]),
            output_matrix=np.array([[2.0, -0.2]]),
            time_step=0.25,
            zero_lag_kernel=np.array([[0.3]]),
            fit_l2_relative_error=0.01,
            spectral_radius=0.9,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "radiation_model.npz"
            saved = save_discrete_state_space_radiation_model(model, path)
            loaded = load_discrete_state_space_radiation_model(saved)

        np.testing.assert_allclose(loaded.state_matrix, model.state_matrix)
        np.testing.assert_allclose(loaded.input_matrix, model.input_matrix)
        np.testing.assert_allclose(loaded.output_matrix, model.output_matrix)
        np.testing.assert_allclose(loaded.zero_lag_kernel, model.zero_lag_kernel)
        self.assertAlmostEqual(loaded.time_step, model.time_step)
        self.assertAlmostEqual(loaded.fit_l2_relative_error, model.fit_l2_relative_error)


if __name__ == "__main__":
    unittest.main()
