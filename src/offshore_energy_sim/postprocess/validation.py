"""Reusable validation metrics for response workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from offshore_energy_sim.postprocess.metrics import rmse


def response_error_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    """Compute whole-response error metrics for two retained response arrays."""

    difference = candidate - baseline
    return {
        "shape_candidate": tuple(candidate.shape),
        "shape_baseline": tuple(baseline.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "l2_relative_error": float(np.linalg.norm(difference) / np.linalg.norm(baseline)),
    }


def curve_error_metrics(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    *,
    quantity_prefix: str = "curve",
) -> dict[str, float]:
    """Compute error metrics between two curves on the candidate x-grid."""

    if not np.allclose(candidate_x, baseline_x):
        raise AssertionError("Candidate and baseline x grids differ.")
    difference = candidate_y - baseline_y
    return {
        f"{quantity_prefix}_max_abs_error": float(np.max(np.abs(difference))),
        f"{quantity_prefix}_rmse": rmse(candidate_y, baseline_y),
    }


def interpolated_curve_rmse(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
) -> float:
    """Compute RMSE after interpolating candidate data onto a reference grid."""

    return rmse(reference_y, np.interp(reference_x, candidate_x, candidate_y))


def load_two_column_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column validation curve as ``x, y`` arrays."""

    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]
