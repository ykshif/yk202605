"""Wave and wind spectral density models."""

from __future__ import annotations

import numpy as np


def jonswap_spectrum(
    significant_wave_height: float,
    peak_period: float,
    omega: np.ndarray,
    gamma: float = 3.3,
) -> np.ndarray:
    """Return the JONSWAP spectrum used by the legacy scripts.

    This preserves `wave_spectrum.jonswap`, including the final `2*pi` factor
    used for comparison with the paper results.
    """

    peak_omega = 2 * np.pi / peak_period
    sigma = np.where(omega <= peak_omega, 0.07, 0.09)
    alpha = 0.0624 / (0.230 + 0.0336 * gamma - (0.185 / (1.9 + gamma)))
    beta = np.exp(-((omega - peak_omega) ** 2) / (2 * (sigma**2) * (peak_omega**2)))
    return (
        alpha
        * significant_wave_height**2
        * peak_omega**4
        * omega ** (-5)
        * gamma**beta
        * np.exp(-1.25 * (peak_omega / omega) ** 4)
        * 2
        * np.pi
    )


def wind_speed_power_law(
    reference_speed: float,
    height: float,
    reference_height: float = 10.0,
    alpha: float = 0.125,
) -> float:
    """Adjust wind speed from a reference height using a power law."""

    return abs(reference_speed * (height / reference_height) ** alpha)


def turbulence_intensity_api(height: float) -> float:
    """Return the piecewise turbulence intensity used in `DM_Windload.py`."""

    exponent = -0.125 if height <= 20 else -0.275
    return 0.15 * (height / 20) ** exponent


def api_wind_spectrum(
    reference_speed: float,
    height: float,
    frequencies: np.ndarray,
    alpha: float = 0.125,
) -> np.ndarray:
    """Return the API wind spectrum used by the wind-load workflow."""

    adjusted_speed = wind_speed_power_law(reference_speed, height, alpha=alpha)
    turbulence = turbulence_intensity_api(height)
    peak_frequency = 0.025 * adjusted_speed / height
    return (
        adjusted_speed**2
        * turbulence**2
        / peak_frequency
        * (1 + 1.5 * (frequencies / peak_frequency)) ** (-5 / 3)
    )


def amplitude_from_spectrum(spectrum: np.ndarray, delta_frequency: float) -> np.ndarray:
    """Convert one-sided spectrum samples to harmonic amplitudes."""

    return np.sqrt(2 * spectrum * delta_frequency)
