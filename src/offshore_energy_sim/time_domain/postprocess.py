"""Post-processing helpers for time-domain verification."""

from __future__ import annotations

import numpy as np


def fit_harmonic_amplitude(
    values: np.ndarray,
    time: np.ndarray,
    omega: float,
    *,
    start_time: float | None = None,
) -> np.ndarray:
    """Fit ``real(X * exp(-1j*omega*t))`` to a time history.

    Returns the complex amplitude ``X``. For a matrix input with shape
    ``(n_time, ndof)``, the output has shape ``(ndof,)``.
    """

    time = np.asarray(time, dtype=float).reshape(-1)
    data = np.asarray(values, dtype=float)
    if data.shape[0] != time.size:
        raise ValueError("values first axis must match time")
    if omega < 0.0:
        raise ValueError("omega must be non-negative")
    if start_time is not None:
        mask = time >= start_time
        if np.count_nonzero(mask) < 3:
            raise ValueError("not enough samples after start_time")
        time = time[mask]
        data = data[mask]

    design = np.column_stack([np.cos(omega * time), np.sin(omega * time)])
    coeffs, *_ = np.linalg.lstsq(design, data, rcond=None)
    return coeffs[0] + 1j * coeffs[1]


def harmonic_amplitude_error(
    fitted: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Return absolute and relative complex-amplitude errors."""

    fitted = np.asarray(fitted, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    delta = fitted - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "l2_abs_error": float(np.linalg.norm(delta)),
        "l2_relative_error": float(np.linalg.norm(delta) / max(reference_norm, 1.0e-30)),
    }


def fit_multi_harmonic_amplitudes(
    values: np.ndarray,
    time: np.ndarray,
    omega: np.ndarray,
    *,
    start_time: float | None = None,
) -> np.ndarray:
    """Fit ``sum real(X_j * exp(-1j*omega_j*t))`` to time histories."""

    time = np.asarray(time, dtype=float).reshape(-1)
    frequencies = np.asarray(omega, dtype=float).reshape(-1)
    data = np.asarray(values, dtype=float)
    if data.shape[0] != time.size:
        raise ValueError("values first axis must match time")
    if frequencies.size == 0:
        raise ValueError("omega must contain at least one frequency")
    if np.any(frequencies < 0.0):
        raise ValueError("omega must be non-negative")
    if start_time is not None:
        mask = time >= start_time
        if np.count_nonzero(mask) < 2 * frequencies.size + 1:
            raise ValueError("not enough samples after start_time")
        time = time[mask]
        data = data[mask]
    if data.ndim == 1:
        data = data[:, np.newaxis]

    phase = np.outer(time, frequencies)
    design = np.column_stack([np.cos(phase), np.sin(phase)])
    coeffs, *_ = np.linalg.lstsq(design, data, rcond=None)
    real = coeffs[: frequencies.size]
    imag = coeffs[frequencies.size :]
    fitted = real + 1j * imag
    return fitted[:, 0] if np.asarray(values).ndim == 1 else fitted


def harmonic_component_variance(amplitudes: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Return variance represented by harmonic complex amplitudes."""

    values = np.asarray(amplitudes)
    return 0.5 * np.sum(np.abs(values) ** 2, axis=axis)


def zero_mean_rms(values: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Return RMS after removing the mean along ``axis``."""

    data = np.asarray(values, dtype=float)
    centered = data - np.mean(data, axis=axis, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=axis))


def relative_l2_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return an L2 relative error with a tiny zero-denominator guard."""

    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))
