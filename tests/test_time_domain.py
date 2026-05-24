from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import (  # noqa: E402
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)
from offshore_energy_sim.reduction import reduce_matrix_dofs  # noqa: E402
from offshore_energy_sim.solver import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.time_domain import (  # noqa: E402
    TimeDomainSimulationConfig,
    apply_radiation_frequency_window,
    direct_convolution_memory_force,
    estimate_infinite_frequency_added_mass,
    estimate_infinite_frequency_added_mass_from_irf,
    external_force_time_series,
    fit_harmonic_amplitude,
    fit_multi_harmonic_amplitudes,
    harmonic_force_time_series,
    harmonic_component_variance,
    jonswap_spectrum,
    prepare_rodm_time_domain_hydrodynamic_terms,
    project_symmetric_positive_semidefinite,
    radiation_coefficients_from_discrete_irf,
    radiation_coefficients_from_irf,
    radiation_frequency_window_weights,
    radiation_irf_from_damping,
    solve_linear_time_domain,
    solve_linear_time_domain_rk4,
    spectral_wave_amplitudes,
    spectral_wave_force_time_series,
    wave_elevation_time_series,
    zero_mean_rms,
)


class TimeDomainTests(unittest.TestCase):
    def test_time_domain_config_validates_radiation_model(self) -> None:
        with self.assertRaises(ValueError):
            TimeDomainSimulationConfig(
                time_step=0.1,
                duration=1.0,
                radiation_model="unsupported",
            )

    def test_psd_projection_clips_negative_eigenvalues(self) -> None:
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

        projected = project_symmetric_positive_semidefinite(matrix)

        eigenvalues = np.linalg.eigvalsh(projected)
        self.assertGreaterEqual(float(eigenvalues[0]), -1.0e-12)
        np.testing.assert_allclose(projected, projected.T)

    def test_radiation_frequency_window_weights_taper_tail(self) -> None:
        omega = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

        linear = radiation_frequency_window_weights(
            omega,
            window="linear_tail",
            start_omega=1.0,
            stop_omega=2.0,
        )
        cosine = radiation_frequency_window_weights(
            omega,
            window="cosine_tail",
            start_omega=1.0,
            stop_omega=2.0,
        )

        np.testing.assert_allclose(linear, [1.0, 1.0, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(cosine, [1.0, 1.0, 1.0, 0.5, 0.0])

    def test_apply_radiation_frequency_window_scales_damping_series(self) -> None:
        omega = np.array([1.0, 2.0, 3.0])
        damping = np.ones((3, 2, 2))

        tapered = apply_radiation_frequency_window(
            omega,
            damping,
            window="linear_tail",
            start_omega=1.0,
            stop_omega=3.0,
        )

        np.testing.assert_allclose(tapered[:, 0, 0], [1.0, 0.5, 0.0])

    def test_jonswap_spectrum_normalizes_to_significant_wave_height(self) -> None:
        omega = np.linspace(0.2, 3.0, 200)
        hs = 2.4

        spectrum = jonswap_spectrum(
            omega,
            significant_wave_height=hs,
            peak_period=8.0,
            gamma=3.3,
        )

        np.testing.assert_allclose(np.trapz(spectrum, omega), hs**2 / 16.0)

    def test_spectral_wave_force_matches_single_harmonic_limit(self) -> None:
        omega = np.array([1.0, 2.0])
        density = np.array([0.0, 0.0])
        amplitudes = spectral_wave_amplitudes(omega, density)
        amplitudes[0] = 2.0
        phases = np.array([0.25, 0.0])
        time = np.array([0.0, 0.4])
        force_hat = np.array([[3.0 + 4.0j], [100.0 + 0.0j]])

        force = spectral_wave_force_time_series(force_hat, omega, time, amplitudes, phases)
        expected = harmonic_force_time_series(
            force_hat[0],
            omega[0],
            time,
            amplitude=2.0,
            phase_rad=0.25,
        )

        np.testing.assert_allclose(force, expected)

    def test_external_force_time_series_interpolates_columns(self) -> None:
        source_time = np.array([0.0, 1.0, 2.0])
        source_force = np.array([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]])
        target_time = np.array([0.5, 1.5])

        interpolated = external_force_time_series(source_time, source_force, target_time)

        np.testing.assert_allclose(interpolated, [[1.0, 11.0], [3.0, 13.0]])

    def test_multi_harmonic_fit_recovers_component_amplitudes(self) -> None:
        omega = np.array([0.7, 1.4])
        amplitudes = np.array([1.2 * np.exp(-0.3j), 0.5 * np.exp(0.8j)])
        time = np.linspace(0.0, 80.0, 1000)
        values = np.real(
            amplitudes[0] * np.exp(-1j * omega[0] * time)
            + amplitudes[1] * np.exp(-1j * omega[1] * time)
        )

        fitted = fit_multi_harmonic_amplitudes(values, time, omega)

        np.testing.assert_allclose(fitted, amplitudes, rtol=1.0e-3, atol=1.0e-3)

    def test_wave_elevation_fit_recovers_random_component_phases(self) -> None:
        omega = np.array([0.6, 1.1, 1.7])
        amplitudes = np.array([0.5, 0.25, 0.15])
        phases = np.array([0.2, 1.1, 2.2])
        time = np.linspace(0.0, 120.0, 1500)

        elevation = wave_elevation_time_series(omega, amplitudes, phases, time)
        fitted = fit_multi_harmonic_amplitudes(elevation, time, omega)

        np.testing.assert_allclose(fitted, amplitudes * np.exp(-1j * phases), rtol=1.0e-3, atol=1.0e-3)

    def test_harmonic_component_variance_matches_expected_rms(self) -> None:
        amplitudes = np.array([2.0, 3.0j])

        variance = harmonic_component_variance(amplitudes)

        np.testing.assert_allclose(variance, 0.5 * (4.0 + 9.0))

    def test_zero_mean_rms_removes_bias(self) -> None:
        values = np.array([2.0, 4.0, 6.0])

        np.testing.assert_allclose(zero_mean_rms(values), np.sqrt(8.0 / 3.0))

    def test_harmonic_force_uses_existing_frequency_domain_phase_convention(self) -> None:
        time = np.array([0.0, np.pi / 4.0])
        force = harmonic_force_time_series(
            np.array([1.0 + 2.0j]),
            omega=1.0,
            time=time,
        )

        expected = np.array(
            [
                [1.0],
                [1.0 * np.cos(np.pi / 4.0) + 2.0 * np.sin(np.pi / 4.0)],
            ]
        )
        np.testing.assert_allclose(force, expected)

    def test_newmark_harmonic_steady_state_matches_frequency_solver(self) -> None:
        mass = np.array([[2.0]])
        damping = np.array([[2.0]])
        stiffness = np.array([[40.0]])
        force_hat = np.array([[3.0 + 2.0j]])
        omega = 1.3
        period = 2.0 * np.pi / omega
        dt = period / 240.0
        time = np.arange(0.0, 80.0 * period + 0.5 * dt, dt)
        force = harmonic_force_time_series(force_hat.reshape(-1), omega, time)

        result = solve_linear_time_domain(mass, damping, stiffness, force, time)
        fitted = fit_harmonic_amplitude(
            result.displacement,
            result.time,
            omega,
            start_time=55.0 * period,
        )
        reference = solve_frequency_domain(mass, damping, stiffness, force_hat, omega).reshape(-1)

        np.testing.assert_allclose(fitted, reference, rtol=2.0e-3, atol=2.0e-4)

    def test_rk4_harmonic_steady_state_matches_frequency_solver(self) -> None:
        mass = np.array([[2.0]])
        damping = np.array([[2.0]])
        stiffness = np.array([[40.0]])
        force_hat = np.array([[3.0 + 2.0j]])
        omega = 1.3
        period = 2.0 * np.pi / omega
        dt = period / 240.0
        time = np.arange(0.0, 80.0 * period + 0.5 * dt, dt)
        force = harmonic_force_time_series(force_hat.reshape(-1), omega, time)

        result = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            integrator="rk4",
        )
        fitted = fit_harmonic_amplitude(
            result.displacement,
            result.time,
            omega,
            start_time=55.0 * period,
        )
        reference = solve_frequency_domain(mass, damping, stiffness, force_hat, omega).reshape(-1)

        np.testing.assert_allclose(fitted, reference, rtol=2.0e-3, atol=2.0e-4)

    def test_rk4_wrapper_matches_explicit_rk4_solver(self) -> None:
        mass = np.array([[1.2]])
        damping = np.array([[0.3]])
        stiffness = np.array([[6.0]])
        time = np.linspace(0.0, 2.0, 201)
        force = np.sin(1.7 * time)[:, np.newaxis]

        wrapped = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            integrator="rk4",
        )
        explicit = solve_linear_time_domain_rk4(mass, damping, stiffness, force, time)

        np.testing.assert_allclose(wrapped.displacement, explicit.displacement)
        np.testing.assert_allclose(wrapped.velocity, explicit.velocity)
        np.testing.assert_allclose(wrapped.acceleration, explicit.acceleration)

    def test_linear_time_solver_rejects_unknown_integrator(self) -> None:
        with self.assertRaises(ValueError):
            solve_linear_time_domain(
                np.eye(1),
                np.zeros((1, 1)),
                np.eye(1),
                np.zeros((2, 1)),
                np.array([0.0, 0.1]),
                integrator="bad",
            )

    def test_zero_radiation_irf_matches_plain_linear_time_solver(self) -> None:
        mass = np.array([[1.5]])
        damping = np.array([[0.2]])
        stiffness = np.array([[8.0]])
        time = np.linspace(0.0, 3.0, 61)
        force = np.cos(time)[:, np.newaxis]
        zero_irf = np.zeros((4, 1, 1))

        plain = solve_linear_time_domain(mass, damping, stiffness, force, time)
        memory = solve_linear_time_domain(
            mass,
            damping,
            stiffness,
            force,
            time,
            radiation_irf=zero_irf,
        )

        np.testing.assert_allclose(memory.displacement, plain.displacement)
        np.testing.assert_allclose(memory.velocity, plain.velocity)
        np.testing.assert_allclose(memory.acceleration, plain.acceleration)
        np.testing.assert_allclose(memory.memory_force, 0.0)

    def test_radiation_irf_matches_analytic_cosine_transform(self) -> None:
        omega = np.linspace(0.0, 80.0, 4001)
        time = np.linspace(0.0, 4.0, 17)
        alpha = 1.7
        base = np.array([[2.0, 0.25], [0.25, 1.0]])
        damping = np.exp(-alpha * omega)[:, np.newaxis, np.newaxis] * base

        irf = radiation_irf_from_damping(omega, damping, time)
        analytic = (
            (2.0 / np.pi)
            * (alpha / (alpha**2 + time[:, np.newaxis, np.newaxis] ** 2))
            * base[np.newaxis, :, :]
        )

        np.testing.assert_allclose(irf, analytic, rtol=2.0e-4, atol=2.0e-4)

    def test_direct_convolution_uses_previous_velocity_history(self) -> None:
        velocity = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        kernel = np.array(
            [
                [[100.0, 0.0], [0.0, 100.0]],
                [[2.0, 0.5], [0.0, 1.0]],
                [[1.0, 0.0], [0.25, 0.5]],
            ]
        )

        force = direct_convolution_memory_force(
            velocity,
            kernel,
            time_step=0.1,
            step_index=3,
        )

        expected = 0.1 * (kernel[1] @ velocity[2] + kernel[2] @ velocity[1])
        np.testing.assert_allclose(force, expected)

    def test_trapezoidal_convolution_half_weights_oldest_history(self) -> None:
        velocity = np.array([[0.0], [2.0], [4.0], [6.0]])
        kernel = np.array([[[100.0]], [[3.0]], [[5.0]]])

        force = direct_convolution_memory_force(
            velocity,
            kernel,
            time_step=0.2,
            step_index=3,
            convolution_rule="trapezoidal",
        )

        expected = 0.2 * (kernel[1] @ velocity[2] + 0.5 * kernel[2] @ velocity[1])
        np.testing.assert_allclose(force, expected.reshape(-1))

    def test_infinite_added_mass_uses_high_frequency_tail(self) -> None:
        omega = np.array([0.5, 1.0, 1.5, 2.0])
        mass = np.array([value * np.eye(2) for value in (1.0, 2.0, 5.0, 7.0)])

        actual = estimate_infinite_frequency_added_mass(omega, mass, tail_count=2)

        np.testing.assert_allclose(actual, 6.0 * np.eye(2))

    def test_irf_recovers_cummins_radiation_coefficients(self) -> None:
        time = np.linspace(0.0, 80.0, 8001)
        omega = np.array([0.5, 1.3, 2.1])
        decay = 1.4
        scale = 2.5
        base = np.array([[2.0, 0.3], [0.3, 1.0]])
        irf = scale * np.exp(-decay * time)[:, np.newaxis, np.newaxis] * base
        a_inf = 7.0 * base
        expected_damping = (
            scale
            * decay
            / (decay**2 + omega[:, np.newaxis, np.newaxis] ** 2)
            * base[np.newaxis, :, :]
        )
        expected_added = (
            a_inf[np.newaxis, :, :]
            - scale
            / (decay**2 + omega[:, np.newaxis, np.newaxis] ** 2)
            * base[np.newaxis, :, :]
        )

        added, damping = radiation_coefficients_from_irf(
            omega,
            irf,
            time,
            added_mass_infinite=a_inf,
        )

        np.testing.assert_allclose(damping, expected_damping, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(added, expected_added, rtol=2.0e-4, atol=2.0e-4)

    def test_ogilvie_added_mass_estimate_recovers_infinite_mass(self) -> None:
        time = np.linspace(0.0, 80.0, 8001)
        omega = np.array([0.5, 1.3, 2.1])
        decay = 1.4
        scale = 2.5
        base = np.array([[1.0, 0.1], [0.1, 0.7]])
        irf = scale * np.exp(-decay * time)[:, np.newaxis, np.newaxis] * base
        a_inf = 5.0 * base
        added = (
            a_inf[np.newaxis, :, :]
            - scale
            / (decay**2 + omega[:, np.newaxis, np.newaxis] ** 2)
            * base[np.newaxis, :, :]
        )

        actual = estimate_infinite_frequency_added_mass_from_irf(omega, added, irf, time)

        np.testing.assert_allclose(actual, a_inf, rtol=2.0e-4, atol=2.0e-4)

    def test_rodm_direct_convolution_hydrodynamic_preprocessing(self) -> None:
        case = RodmFrequencyCase(
            case_id="synthetic",
            total_nodes=2,
            full_dofs_per_node=3,
            retained_dofs_per_node=2,
            removed_full_dofs_zero_based=(2,),
            master_node_rule=MasterNodeRule(first_node=1, node_interval=1, count=2),
            hydrodynamic_dataset=Path("unused.nc"),
            structural_matrices=StructuralMatrixPaths(
                mass=Path("unused_mass.mtx"),
                stiffness=Path("unused_stiffness.mtx"),
            ),
            hydrodynamic_nodes=2,
            hydrodynamic_dof_to_remove_zero_based=2,
            frequency_index=0,
        )
        omega = np.array([2.0, 1.0, 3.0])
        added = np.array([20.0 * np.eye(6), 10.0 * np.eye(6), 30.0 * np.eye(6)])
        damping = np.array([2.0 * np.eye(6), 1.0 * np.eye(6), 3.0 * np.eye(6)])
        hydrostatic = 4.0 * np.eye(6)
        froude_krylov = np.arange(18, dtype=float).reshape(3, 6) + 1.0j
        diffraction = 0.5 * np.ones((3, 6), dtype=complex)
        dataset = xr.Dataset(
            data_vars={
                "added_mass": (("omega", "row", "col"), added),
                "radiation_damping": (("omega", "row", "col"), damping),
                "hydrostatic_stiffness": (("row", "col"), hydrostatic),
                "Froude_Krylov_force": (("omega", "dof"), froude_krylov),
                "diffraction_force": (("omega", "dof"), diffraction),
            },
            coords={"omega": omega},
        )
        config = TimeDomainSimulationConfig(
            time_step=0.25,
            duration=4.0,
            radiation_model="direct_convolution",
            memory_duration=1.0,
            added_mass_tail_count=1,
        )

        terms = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)

        np.testing.assert_allclose(terms.omega_grid, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(terms.added_mass, 20.0 * np.eye(4))
        np.testing.assert_allclose(terms.added_mass_infinite, 30.0 * np.eye(4))
        self.assertEqual(terms.radiation_irf.shape, (5, 4, 4))
        self.assertEqual(terms.radiation_irf_time.tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            terms.hydrostatic_stiffness,
            reduce_matrix_dofs(hydrostatic, 2, [2]),
        )
        np.testing.assert_allclose(
            terms.wave_force.reshape(-1),
            np.array([0.5 + 1.0j, 1.5 + 1.0j, 3.5 + 1.0j, 4.5 + 1.0j]),
        )

    def test_selected_frequency_residual_matches_irf_coefficients(self) -> None:
        case = RodmFrequencyCase(
            case_id="synthetic",
            total_nodes=1,
            full_dofs_per_node=3,
            retained_dofs_per_node=2,
            removed_full_dofs_zero_based=(2,),
            master_node_rule=MasterNodeRule(first_node=1, node_interval=1, count=1),
            hydrodynamic_dataset=Path("unused.nc"),
            structural_matrices=StructuralMatrixPaths(
                mass=Path("unused_mass.mtx"),
                stiffness=Path("unused_stiffness.mtx"),
            ),
            hydrodynamic_nodes=1,
            hydrodynamic_dof_to_remove_zero_based=2,
            frequency_index=1,
        )
        omega = np.array([0.5, 1.0, 1.5])
        added = np.array([2.0 * np.eye(3), 4.0 * np.eye(3), 8.0 * np.eye(3)])
        damping = np.array([0.2 * np.eye(3), 0.4 * np.eye(3), 0.8 * np.eye(3)])
        dataset = xr.Dataset(
            data_vars={
                "added_mass": (("omega", "row", "col"), added),
                "radiation_damping": (("omega", "row", "col"), damping),
                "hydrostatic_stiffness": (("row", "col"), np.eye(3)),
                "Froude_Krylov_force": (("omega", "dof"), np.ones((3, 3), dtype=complex)),
                "diffraction_force": (("omega", "dof"), np.zeros((3, 3), dtype=complex)),
            },
            coords={"omega": omega},
        )
        config = TimeDomainSimulationConfig(
            time_step=0.1,
            duration=2.0,
            radiation_model="direct_convolution",
            memory_duration=0.5,
            added_mass_tail_count=1,
            radiation_residual_model="selected_frequency",
        )

        terms = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)
        reconstructed_added, reconstructed_damping = radiation_coefficients_from_irf(
            terms.omega,
            terms.radiation_irf,
            terms.radiation_irf_time,
            added_mass_infinite=terms.added_mass_infinite,
        )
        discrete_added, discrete_damping = radiation_coefficients_from_discrete_irf(
            terms.omega,
            terms.radiation_irf,
            terms.radiation_irf_time,
            added_mass_infinite=terms.added_mass_infinite,
            convolution_rule=config.radiation_convolution_rule,
        )

        self.assertGreater(np.linalg.norm(reconstructed_damping - discrete_damping), 0.0)
        np.testing.assert_allclose(discrete_added + terms.residual_added_mass, terms.added_mass)
        np.testing.assert_allclose(
            discrete_damping + terms.residual_radiation_damping,
            terms.radiation_damping,
        )


if __name__ == "__main__":
    unittest.main()
