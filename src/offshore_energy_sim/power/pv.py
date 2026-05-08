"""PV power and power-loss helper models."""

from __future__ import annotations

import numpy as np


def dc_power_from_irradiance(
    irradiance_w_m2: np.ndarray | float,
    panel_area_m2: np.ndarray | float,
    efficiency: float,
) -> np.ndarray:
    """Return ideal DC power from irradiance, area, and efficiency."""

    return np.asarray(irradiance_w_m2) * np.asarray(panel_area_m2) * efficiency


def cosine_incidence_factor(tilt_rad: np.ndarray | float) -> np.ndarray:
    """Return clipped cosine loss factor for panel tilt relative to incoming light."""

    return np.clip(np.cos(tilt_rad), 0.0, 1.0)


def power_with_tilt_loss(
    irradiance_w_m2: np.ndarray | float,
    panel_area_m2: np.ndarray | float,
    efficiency: float,
    tilt_rad: np.ndarray | float,
) -> np.ndarray:
    """Return PV power after a simple cosine tilt/incidence loss."""

    return dc_power_from_irradiance(irradiance_w_m2, panel_area_m2, efficiency) * cosine_incidence_factor(
        tilt_rad
    )


def relative_power_loss(reference_power: np.ndarray, actual_power: np.ndarray) -> np.ndarray:
    """Return relative power loss, guarding against zero reference power."""

    reference_power = np.asarray(reference_power)
    actual_power = np.asarray(actual_power)
    return np.divide(
        reference_power - actual_power,
        reference_power,
        out=np.zeros_like(actual_power, dtype=float),
        where=reference_power != 0,
    )
