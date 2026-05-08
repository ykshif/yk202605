"""Response spectrum postprocessing helpers."""

from __future__ import annotations

import numpy as np


def response_spectrum_from_amplitude(
    response_amplitude: np.ndarray,
    delta_frequency: float,
) -> np.ndarray:
    """Convert response amplitude samples to a response spectrum estimate."""

    return np.asarray(response_amplitude) ** 2 / delta_frequency


def rms_from_spectrum(
    spectrum: np.ndarray,
    delta_frequency: float,
    axis: int | None = None,
) -> np.ndarray:
    """Return RMS response from spectral density samples."""

    return np.sqrt(np.sum(np.asarray(spectrum) * delta_frequency, axis=axis))
