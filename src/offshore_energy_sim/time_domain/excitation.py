"""Time-domain excitation-force generation."""

from __future__ import annotations

import numpy as np


def cosine_ramp(time: np.ndarray, ramp_time: float) -> np.ndarray:
    """Return a smooth startup ramp from 0 to 1.

    The ramp is one for all samples when ``ramp_time <= 0``. This is useful for
    regular-wave verification because it reduces transient free vibration
    without changing the target steady-state harmonic force.
    """

    time = np.asarray(time, dtype=float)
    if ramp_time <= 0.0:
        return np.ones_like(time)
    ramp = np.ones_like(time)
    active = time < ramp_time
    ramp[active] = 0.5 * (1.0 - np.cos(np.pi * time[active] / ramp_time))
    return ramp


def harmonic_force_time_series(
    force_hat: np.ndarray,
    omega: float,
    time: np.ndarray,
    *,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
    ramp_time: float = 0.0,
    convention: str = "legacy_negative",
) -> np.ndarray:
    """Convert a complex frequency-domain force to a real time series.

    The existing frequency-domain solver uses
    ``-omega**2 M - 1j*omega C + K``. That corresponds to the harmonic
    convention ``real(F_hat * exp(-1j*omega*t))``. The default therefore uses
    ``legacy_negative`` so a steady-state time-domain response can be compared
    directly with ``solve_frequency_domain``.
    """

    if omega < 0.0:
        raise ValueError("omega must be non-negative")
    force = np.asarray(force_hat, dtype=np.complex128).reshape(-1)
    time = np.asarray(time, dtype=float)
    if convention == "legacy_negative":
        phase = np.exp(-1j * (omega * time + phase_rad))
    elif convention == "positive":
        phase = np.exp(1j * (omega * time + phase_rad))
    else:
        raise ValueError("convention must be 'legacy_negative' or 'positive'")
    values = amplitude * np.real(phase[:, np.newaxis] * force[np.newaxis, :])
    return cosine_ramp(time, ramp_time)[:, np.newaxis] * values


def spectral_component_widths(omega: np.ndarray) -> np.ndarray:
    """Return frequency-bin widths for an increasing angular-frequency grid."""

    values = np.asarray(omega, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("omega must contain at least two values")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = max(0.0, values[0] - 0.5 * (values[1] - values[0]))
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.diff(edges)


def normalize_wave_spectrum(
    omega: np.ndarray,
    shape: np.ndarray,
    *,
    significant_wave_height: float,
) -> np.ndarray:
    """Scale a nonnegative spectral shape to the requested ``H_s``."""

    if significant_wave_height < 0.0:
        raise ValueError("significant_wave_height must be non-negative")
    omega = np.asarray(omega, dtype=float).reshape(-1)
    values = np.asarray(shape, dtype=float).reshape(-1)
    if values.shape != omega.shape:
        raise ValueError("shape must have the same length as omega")
    if np.any(values < 0.0):
        raise ValueError("spectrum shape must be non-negative")
    if significant_wave_height == 0.0:
        return np.zeros_like(values)
    moment = float(np.trapz(values, omega))
    if moment <= 0.0:
        raise ValueError("spectrum shape has zero area")
    target_m0 = significant_wave_height**2 / 16.0
    return values * (target_m0 / moment)


def pierson_moskowitz_spectrum(
    omega: np.ndarray,
    *,
    significant_wave_height: float,
    peak_period: float,
) -> np.ndarray:
    """Return a PM spectrum in angular-frequency form normalized to ``H_s``."""

    if peak_period <= 0.0:
        raise ValueError("peak_period must be positive")
    omega = np.asarray(omega, dtype=float).reshape(-1)
    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive")
    omega_peak = 2.0 * np.pi / peak_period
    shape = omega**-5 * np.exp(-1.25 * (omega_peak / omega) ** 4)
    return normalize_wave_spectrum(
        omega,
        shape,
        significant_wave_height=significant_wave_height,
    )


def jonswap_spectrum(
    omega: np.ndarray,
    *,
    significant_wave_height: float,
    peak_period: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """Return a JONSWAP spectrum normalized numerically to ``H_s``."""

    if peak_period <= 0.0:
        raise ValueError("peak_period must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    omega = np.asarray(omega, dtype=float).reshape(-1)
    if np.any(omega <= 0.0):
        raise ValueError("omega must be positive")
    omega_peak = 2.0 * np.pi / peak_period
    sigma = np.where(omega <= omega_peak, 0.07, 0.09)
    exponent = np.exp(-((omega / omega_peak - 1.0) ** 2) / (2.0 * sigma**2))
    shape = omega**-5 * np.exp(-1.25 * (omega_peak / omega) ** 4) * gamma**exponent
    return normalize_wave_spectrum(
        omega,
        shape,
        significant_wave_height=significant_wave_height,
    )


def wave_spectrum_density(
    omega: np.ndarray,
    *,
    spectrum_type: str,
    significant_wave_height: float,
    peak_period: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """Return a normalized wave spectrum for a named spectrum type."""

    spectrum = str(spectrum_type).lower()
    if spectrum == "pierson_moskowitz":
        return pierson_moskowitz_spectrum(
            omega,
            significant_wave_height=significant_wave_height,
            peak_period=peak_period,
        )
    if spectrum == "jonswap":
        return jonswap_spectrum(
            omega,
            significant_wave_height=significant_wave_height,
            peak_period=peak_period,
            gamma=gamma,
        )
    raise ValueError("spectrum_type must be 'jonswap' or 'pierson_moskowitz'")


def spectral_wave_amplitudes(omega: np.ndarray, spectrum_density: np.ndarray) -> np.ndarray:
    """Return component wave amplitudes from ``S(omega)``."""

    density = np.asarray(spectrum_density, dtype=float).reshape(-1)
    widths = spectral_component_widths(omega)
    if density.shape != widths.shape:
        raise ValueError("spectrum_density must have the same length as omega")
    if np.any(density < 0.0):
        raise ValueError("spectrum_density must be non-negative")
    return np.sqrt(2.0 * density * widths)


def random_wave_phases(component_count: int, *, seed: int | None = None) -> np.ndarray:
    """Return repeatable random phases for spectral-wave components."""

    if component_count < 1:
        raise ValueError("component_count must be positive")
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, int(component_count))


def wave_elevation_time_series(
    omega: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    time: np.ndarray,
    *,
    ramp_time: float = 0.0,
    convention: str = "legacy_negative",
) -> np.ndarray:
    """Synthesize a real wave-elevation time series from components."""

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    amp = np.asarray(amplitudes, dtype=float).reshape(-1)
    phase_values = np.asarray(phases, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    if amp.shape != omega_values.shape or phase_values.shape != omega_values.shape:
        raise ValueError("amplitudes and phases must match omega")
    if convention == "legacy_negative":
        harmonic = np.exp(-1j * (time[:, np.newaxis] * omega_values[np.newaxis, :] + phase_values[np.newaxis, :]))
    elif convention == "positive":
        harmonic = np.exp(1j * (time[:, np.newaxis] * omega_values[np.newaxis, :] + phase_values[np.newaxis, :]))
    else:
        raise ValueError("convention must be 'legacy_negative' or 'positive'")
    values = np.real(harmonic * amp[np.newaxis, :]).sum(axis=1)
    return cosine_ramp(time, ramp_time) * values


def spectral_wave_force_time_series(
    force_hat_by_omega: np.ndarray,
    omega: np.ndarray,
    time: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    *,
    ramp_time: float = 0.0,
    convention: str = "legacy_negative",
) -> np.ndarray:
    """Synthesize excitation-force histories from BEM force transfer functions."""

    force = np.asarray(force_hat_by_omega, dtype=np.complex128)
    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    amp = np.asarray(amplitudes, dtype=float).reshape(-1)
    phase_values = np.asarray(phases, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    if force.ndim != 2:
        raise ValueError("force_hat_by_omega must have shape (n_omega, ndof)")
    if force.shape[0] != omega_values.size:
        raise ValueError("force_hat_by_omega first axis must match omega")
    if amp.shape != omega_values.shape or phase_values.shape != omega_values.shape:
        raise ValueError("amplitudes and phases must match omega")
    if convention == "legacy_negative":
        harmonic = np.exp(-1j * (time[:, np.newaxis] * omega_values[np.newaxis, :] + phase_values[np.newaxis, :]))
    elif convention == "positive":
        harmonic = np.exp(1j * (time[:, np.newaxis] * omega_values[np.newaxis, :] + phase_values[np.newaxis, :]))
    else:
        raise ValueError("convention must be 'legacy_negative' or 'positive'")
    weighted_force = amp[:, np.newaxis] * force
    values = np.real(harmonic @ weighted_force)
    return cosine_ramp(time, ramp_time)[:, np.newaxis] * values


def external_force_time_series(
    source_time: np.ndarray,
    source_force: np.ndarray,
    target_time: np.ndarray,
    *,
    expected_dofs: int | None = None,
) -> np.ndarray:
    """Interpolate a user-supplied external force history onto solver time."""

    source_time = np.asarray(source_time, dtype=float).reshape(-1)
    source_force = np.asarray(source_force, dtype=float)
    target_time = np.asarray(target_time, dtype=float).reshape(-1)
    if source_time.size < 2:
        raise ValueError("source_time must contain at least two samples")
    if np.any(np.diff(source_time) <= 0.0):
        raise ValueError("source_time must be strictly increasing")
    if source_force.ndim != 2 or source_force.shape[0] != source_time.size:
        raise ValueError("source_force must have shape (n_time, ndof)")
    if expected_dofs is not None and source_force.shape[1] != expected_dofs:
        raise ValueError(f"source_force must have {expected_dofs} DOF columns")
    columns = [
        np.interp(target_time, source_time, source_force[:, index])
        for index in range(source_force.shape[1])
    ]
    return np.stack(columns, axis=1)
