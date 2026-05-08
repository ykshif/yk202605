"""Postprocessing metrics for validation and regression checks."""

from __future__ import annotations

import numpy as np


def rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean square error between two equally shaped arrays."""

    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))
