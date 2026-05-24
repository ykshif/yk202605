"""Validation helpers for WEC-Sim-like adapter studies."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def zero_mean_rms(values: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Return RMS after removing the mean along one axis."""

    data = np.asarray(values, dtype=float)
    centered = data - np.mean(data, axis=axis, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=axis))


def time_series_drift_metrics(values: np.ndarray) -> dict[str, float]:
    """Return simple drift indicators for a time-major response array."""

    data = np.asarray(values, dtype=float)
    if data.ndim < 1 or data.shape[0] < 2:
        raise ValueError("values must contain at least two time samples")
    early_count = max(1, data.shape[0] // 10)
    late_count = max(1, data.shape[0] // 10)
    early_mean = np.mean(data[:early_count], axis=0)
    late_mean = np.mean(data[-late_count:], axis=0)
    full_rms = zero_mean_rms(data, axis=0)
    drift = late_mean - early_mean
    return {
        "mean_drift_l2": float(np.linalg.norm(drift)),
        "mean_drift_max_abs": float(np.max(np.abs(drift))),
        "mean_drift_over_rms_l2": float(np.linalg.norm(drift) / max(float(np.linalg.norm(full_rms)), 1.0e-30)),
        "mean_drift_over_rms_max": float(
            np.max(np.abs(drift)) / max(float(np.max(np.abs(full_rms))), 1.0e-30)
        ),
    }


def load_case_statistics(case_root: str | Path) -> dict[str, object]:
    """Load time-domain case metrics plus derived memory-force and drift metrics."""

    root = Path(case_root)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    statistics_path = root / "statistics_validation" / "statistics_metrics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.exists() else {}
    memory = np.load(root / "memory_force_time.npy")
    heave = np.load(root / "centerline_heave_time.npy")
    memory_norm = np.linalg.norm(memory, axis=1)
    derived = {
        "radiation_force_rms": float(np.sqrt(np.mean(memory_norm**2))),
        "radiation_force_abs_max": float(np.max(np.abs(memory))),
        "heave_drift": time_series_drift_metrics(heave),
    }
    return {
        "metrics": metrics,
        "statistics": statistics,
        "derived": derived,
    }


def compare_case_statistics(
    before_case_root: str | Path,
    after_case_root: str | Path,
) -> dict[str, object]:
    """Compare original-band and extrapolated-band time-domain validation cases."""

    before = load_case_statistics(before_case_root)
    after = load_case_statistics(after_case_root)

    def metric(path: tuple[str, ...], source: dict[str, object]) -> float | None:
        current: object = source
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return float(current) if current is not None else None

    comparisons: dict[str, dict[str, float | None]] = {}
    for name, path in {
        "wave_variance_closure_error": (
            "statistics",
            "wave_variance_time_vs_component_relative_error",
        ),
        "excitation_rms_closure_error": (
            "statistics",
            "excitation_force_rms_l2_relative_error",
        ),
        "centerline_heave_rms_closure_error": (
            "statistics",
            "centerline_heave_rms_l2_relative_error",
        ),
        "centerline_heave_rms_max": ("metrics", "centerline_heave_rms_max"),
        "radiation_force_rms": ("derived", "radiation_force_rms"),
        "heave_mean_drift_over_rms_l2": ("derived", "heave_drift", "mean_drift_over_rms_l2"),
    }.items():
        before_value = metric(path, before)
        after_value = metric(path, after)
        delta = None if before_value is None or after_value is None else after_value - before_value
        ratio = (
            None
            if before_value is None or after_value is None
            else after_value / max(abs(before_value), 1.0e-30)
        )
        comparisons[name] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "after_over_before": ratio,
        }
    return {
        "before": before,
        "after": after,
        "comparisons": comparisons,
    }
