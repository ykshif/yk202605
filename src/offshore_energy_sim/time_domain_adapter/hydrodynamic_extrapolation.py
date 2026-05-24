"""Hydrodynamic frequency-grid checks and opt-in extrapolation.

The functions here are adapter-layer utilities. They never mutate the original
RODM/BEM arrays; extrapolated values are returned as new arrays with the
original frequency range embedded unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HydrodynamicExtrapolationConfig:
    """Controls for low/high-frequency hydrodynamic extrapolation."""

    low_frequency_min: float | None = None
    high_frequency_max: float | None = None
    low_frequency_count: int = 4
    high_frequency_count: int = 24
    added_mass_tail_count: int = 3
    added_mass_high_power: float = 2.0
    damping_low_power: float = 1.0
    damping_high_power: float = 2.0
    force_high_power: float = 1.0
    high_frequency_taper_to_zero: bool = True

    def __post_init__(self) -> None:
        if self.low_frequency_min is not None and self.low_frequency_min < 0.0:
            raise ValueError("low_frequency_min must be non-negative")
        if self.high_frequency_max is not None and self.high_frequency_max <= 0.0:
            raise ValueError("high_frequency_max must be positive")
        if self.low_frequency_count < 0:
            raise ValueError("low_frequency_count must be non-negative")
        if self.high_frequency_count < 0:
            raise ValueError("high_frequency_count must be non-negative")
        if self.added_mass_tail_count < 1:
            raise ValueError("added_mass_tail_count must be positive")
        for name in (
            "added_mass_high_power",
            "damping_low_power",
            "damping_high_power",
            "force_high_power",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class ExtrapolatedHydrodynamicData:
    """Hydrodynamic arrays on an extended frequency grid."""

    original_omega: np.ndarray
    omega: np.ndarray
    added_mass: np.ndarray
    radiation_damping: np.ndarray
    wave_force: np.ndarray | None
    original_slice: slice
    config: HydrodynamicExtrapolationConfig

    @property
    def original_range_mask(self) -> np.ndarray:
        """Boolean mask selecting the embedded original frequencies."""

        mask = np.zeros(self.omega.size, dtype=bool)
        mask[self.original_slice] = True
        return mask

    def invariance_report(
        self,
        original_added_mass: np.ndarray,
        original_radiation_damping: np.ndarray,
        original_wave_force: np.ndarray | None = None,
    ) -> dict[str, float | None]:
        """Return max absolute changes inside the original frequency range."""

        report: dict[str, float | None] = {
            "added_mass": max_abs_difference_inside_original_range(
                original_added_mass,
                self.added_mass,
                self.original_slice,
            ),
            "radiation_damping": max_abs_difference_inside_original_range(
                original_radiation_damping,
                self.radiation_damping,
                self.original_slice,
            ),
            "wave_force": None,
        }
        if original_wave_force is not None and self.wave_force is not None:
            report["wave_force"] = max_abs_difference_inside_original_range(
                original_wave_force,
                self.wave_force,
                self.original_slice,
            )
        return report


def _as_strictly_increasing_omega(omega: np.ndarray) -> np.ndarray:
    values = np.asarray(omega, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("omega must contain at least two frequencies")
    if not np.all(np.isfinite(values)):
        raise ValueError("omega contains NaN or Inf")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    return values


def frequency_grid_diagnostics(
    omega: np.ndarray,
    *,
    near_zero_threshold: float = 1.0e-3,
    uniform_rtol: float = 1.0e-3,
    reference_omega: float | None = None,
    high_frequency_ratio_threshold: float = 4.0,
) -> dict[str, float | int | bool | None]:
    """Return basic diagnostics for a hydrodynamic frequency grid."""

    values = np.asarray(omega, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    strictly_increasing = bool(values.size >= 2 and np.all(np.diff(values) > 0.0))
    if values.size >= 2 and np.all(finite):
        delta = np.diff(values)
        delta_mean = float(np.mean(delta))
        delta_min = float(np.min(delta))
        delta_max = float(np.max(delta))
        delta_std = float(np.std(delta))
        is_uniform = bool(delta_mean > 0.0 and np.max(np.abs(delta - delta_mean)) <= uniform_rtol * delta_mean)
    else:
        delta_mean = delta_min = delta_max = delta_std = float("nan")
        is_uniform = False

    omega_min = float(np.min(values)) if values.size else float("nan")
    omega_max = float(np.max(values)) if values.size else float("nan")
    high_ratio = None
    high_enough = None
    if reference_omega is not None and reference_omega > 0.0 and np.isfinite(omega_max):
        high_ratio = float(omega_max / reference_omega)
        high_enough = bool(high_ratio >= high_frequency_ratio_threshold)

    return {
        "count": int(values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "omega_min": omega_min,
        "omega_max": omega_max,
        "delta_omega_min": delta_min,
        "delta_omega_max": delta_max,
        "delta_omega_mean": delta_mean,
        "delta_omega_std": delta_std,
        "is_strictly_increasing": strictly_increasing,
        "is_uniform": is_uniform,
        "is_near_zero": bool(values.size > 0 and abs(omega_min) <= near_zero_threshold),
        "reference_omega": None if reference_omega is None else float(reference_omega),
        "omega_max_over_reference": high_ratio,
        "high_frequency_range_sufficient": high_enough,
    }


def _finite_report(name: str, values: np.ndarray) -> dict[str, float | int | bool]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    return {
        f"{name}_nan_count": int(np.count_nonzero(np.isnan(array))),
        f"{name}_inf_count": int(np.count_nonzero(np.isinf(array))),
        f"{name}_finite": bool(np.all(finite)),
        f"{name}_max_abs": float(np.max(np.abs(array))) if array.size else 0.0,
    }


def _roughness_metric(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.shape[0] < 3:
        return 0.0
    second = np.diff(array, n=2, axis=0)
    return float(np.linalg.norm(second.reshape(second.shape[0], -1)) / max(float(np.linalg.norm(array)), 1.0e-30))


def _oscillation_score(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.shape[0] < 4:
        return 0.0
    norm_series = np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)
    first = np.diff(norm_series)
    active = np.abs(first) > 1.0e-12 * max(float(np.max(np.abs(norm_series))), 1.0)
    if np.count_nonzero(active) < 2:
        return 0.0
    signs = np.sign(first[active])
    return float(np.count_nonzero(signs[1:] * signs[:-1] < 0.0) / max(signs.size - 1, 1))


def hydrodynamic_array_diagnostics(
    omega: np.ndarray,
    added_mass: np.ndarray,
    radiation_damping: np.ndarray,
    wave_force: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Check hydrodynamic arrays for finite values and rough frequency trends."""

    values = _as_strictly_increasing_omega(omega)
    added = np.asarray(added_mass, dtype=float)
    damping = np.asarray(radiation_damping, dtype=float)
    if added.shape[0] != values.size:
        raise ValueError("added_mass first axis must match omega")
    if damping.shape[0] != values.size:
        raise ValueError("radiation_damping first axis must match omega")
    if added.ndim != 3 or added.shape[1] != added.shape[2]:
        raise ValueError("added_mass must have shape (n_omega, ndof, ndof)")
    if damping.ndim != 3 or damping.shape[1:] != added.shape[1:]:
        raise ValueError("radiation_damping shape must match added_mass")

    diagnostics: dict[str, float | int | bool] = {}
    diagnostics.update(_finite_report("added_mass", added))
    diagnostics.update(_finite_report("radiation_damping", damping))
    diagnostics["added_mass_roughness"] = _roughness_metric(added)
    diagnostics["radiation_damping_roughness"] = _roughness_metric(damping)
    diagnostics["radiation_damping_oscillation_score"] = _oscillation_score(damping)
    diagnostics["radiation_damping_tail_norm_ratio"] = float(
        np.linalg.norm(damping[-1]) / max(float(np.max(np.linalg.norm(damping.reshape(values.size, -1), axis=1))), 1.0e-30)
    )
    if wave_force is not None:
        force = np.asarray(wave_force)
        if force.shape[0] != values.size:
            raise ValueError("wave_force first axis must match omega")
        diagnostics.update(_finite_report("wave_force", force))
        diagnostics["wave_force_roughness"] = _roughness_metric(force)
        diagnostics["wave_force_oscillation_score"] = _oscillation_score(force)
    return diagnostics


def build_extended_omega_grid(
    omega: np.ndarray,
    config: HydrodynamicExtrapolationConfig,
) -> tuple[np.ndarray, slice]:
    """Return an extended grid and the slice containing the original grid."""

    values = _as_strictly_increasing_omega(omega)
    low = np.empty(0, dtype=float)
    high = np.empty(0, dtype=float)
    if (
        config.low_frequency_min is not None
        and config.low_frequency_count > 0
        and config.low_frequency_min < values[0]
    ):
        low = np.linspace(
            float(config.low_frequency_min),
            float(values[0]),
            int(config.low_frequency_count) + 1,
            dtype=float,
        )[:-1]
    if (
        config.high_frequency_max is not None
        and config.high_frequency_count > 0
        and config.high_frequency_max > values[-1]
    ):
        high = np.linspace(
            float(values[-1]),
            float(config.high_frequency_max),
            int(config.high_frequency_count) + 1,
            dtype=float,
        )[1:]
    extended = np.concatenate([low, values, high])
    original_slice = slice(low.size, low.size + values.size)
    return extended, original_slice


def _high_taper(omega_high: np.ndarray, omega_last: float, omega_stop: float, enabled: bool) -> np.ndarray:
    if not enabled or omega_stop <= omega_last:
        return np.ones_like(omega_high)
    xi = np.clip((omega_high - omega_last) / (omega_stop - omega_last), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * xi))


def _extrapolate_added_mass(
    original_omega: np.ndarray,
    values: np.ndarray,
    extended_omega: np.ndarray,
    original_slice: slice,
    config: HydrodynamicExtrapolationConfig,
) -> np.ndarray:
    result = np.empty((extended_omega.size, *values.shape[1:]), dtype=float)
    result[original_slice] = values
    low_count = original_slice.start
    high_start = original_slice.stop
    if low_count:
        result[:low_count] = values[0]
    if high_start < extended_omega.size:
        tail_count = min(config.added_mass_tail_count, values.shape[0])
        limit = np.mean(values[-tail_count:], axis=0)
        last = values[-1]
        high_omega = extended_omega[high_start:]
        scale = (original_omega[-1] / high_omega) ** config.added_mass_high_power
        result[high_start:] = limit[np.newaxis, ...] + scale[:, np.newaxis, np.newaxis] * (last - limit)[np.newaxis, ...]
    return result


def _extrapolate_damping(
    original_omega: np.ndarray,
    values: np.ndarray,
    extended_omega: np.ndarray,
    original_slice: slice,
    config: HydrodynamicExtrapolationConfig,
) -> np.ndarray:
    result = np.empty((extended_omega.size, *values.shape[1:]), dtype=float)
    result[original_slice] = values
    low_count = original_slice.start
    high_start = original_slice.stop
    if low_count:
        low_omega = extended_omega[:low_count]
        scale = (low_omega / original_omega[0]) ** config.damping_low_power
        result[:low_count] = scale[:, np.newaxis, np.newaxis] * values[0][np.newaxis, ...]
    if high_start < extended_omega.size:
        high_omega = extended_omega[high_start:]
        scale = (original_omega[-1] / high_omega) ** config.damping_high_power
        taper = _high_taper(
            high_omega,
            original_omega[-1],
            extended_omega[-1],
            config.high_frequency_taper_to_zero,
        )
        result[high_start:] = (scale * taper)[:, np.newaxis, np.newaxis] * values[-1][np.newaxis, ...]
    return result


def _extrapolate_force_like(
    original_omega: np.ndarray,
    values: np.ndarray,
    extended_omega: np.ndarray,
    original_slice: slice,
    config: HydrodynamicExtrapolationConfig,
) -> np.ndarray:
    result = np.empty((extended_omega.size, *values.shape[1:]), dtype=values.dtype)
    result[original_slice] = values
    low_count = original_slice.start
    high_start = original_slice.stop
    if low_count:
        result[:low_count] = values[0]
    if high_start < extended_omega.size:
        high_omega = extended_omega[high_start:]
        scale = (original_omega[-1] / high_omega) ** config.force_high_power
        taper = _high_taper(
            high_omega,
            original_omega[-1],
            extended_omega[-1],
            config.high_frequency_taper_to_zero,
        )
        reshape = (high_omega.size,) + (1,) * (values.ndim - 1)
        result[high_start:] = (scale * taper).reshape(reshape) * values[-1]
    return result


def extrapolate_hydrodynamic_data(
    omega: np.ndarray,
    added_mass: np.ndarray,
    radiation_damping: np.ndarray,
    *,
    wave_force: np.ndarray | None = None,
    config: HydrodynamicExtrapolationConfig | None = None,
) -> ExtrapolatedHydrodynamicData:
    """Return hydrodynamic arrays extended outside the original frequency band."""

    cfg = HydrodynamicExtrapolationConfig() if config is None else config
    original_omega = _as_strictly_increasing_omega(omega)
    hydrodynamic_array_diagnostics(original_omega, added_mass, radiation_damping, wave_force)
    extended_omega, original_slice = build_extended_omega_grid(original_omega, cfg)
    added = _extrapolate_added_mass(
        original_omega,
        np.asarray(added_mass, dtype=float),
        extended_omega,
        original_slice,
        cfg,
    )
    damping = _extrapolate_damping(
        original_omega,
        np.asarray(radiation_damping, dtype=float),
        extended_omega,
        original_slice,
        cfg,
    )
    force = None
    if wave_force is not None:
        force = _extrapolate_force_like(
            original_omega,
            np.asarray(wave_force),
            extended_omega,
            original_slice,
            cfg,
        )
    return ExtrapolatedHydrodynamicData(
        original_omega=original_omega.copy(),
        omega=extended_omega,
        added_mass=added,
        radiation_damping=damping,
        wave_force=force,
        original_slice=original_slice,
        config=cfg,
    )


def extrapolate_frequency_series(
    omega: np.ndarray,
    values: np.ndarray,
    *,
    config: HydrodynamicExtrapolationConfig,
    series_kind: str,
) -> tuple[np.ndarray, np.ndarray, slice]:
    """Extrapolate one omega-major series as added mass, damping, or force."""

    original_omega = _as_strictly_increasing_omega(omega)
    extended_omega, original_slice = build_extended_omega_grid(original_omega, config)
    kind = str(series_kind).lower()
    if kind == "added_mass":
        extrapolated = _extrapolate_added_mass(
            original_omega,
            np.asarray(values, dtype=float),
            extended_omega,
            original_slice,
            config,
        )
    elif kind == "radiation_damping":
        extrapolated = _extrapolate_damping(
            original_omega,
            np.asarray(values, dtype=float),
            extended_omega,
            original_slice,
            config,
        )
    elif kind == "force":
        extrapolated = _extrapolate_force_like(
            original_omega,
            np.asarray(values),
            extended_omega,
            original_slice,
            config,
        )
    else:
        raise ValueError("series_kind must be 'added_mass', 'radiation_damping', or 'force'")
    return extended_omega, extrapolated, original_slice


def max_abs_difference_inside_original_range(
    original: np.ndarray,
    extended: np.ndarray,
    original_slice: slice,
) -> float:
    """Return max absolute difference between original and embedded data."""

    original_array = np.asarray(original)
    embedded = np.asarray(extended)[original_slice]
    if embedded.shape != original_array.shape:
        raise ValueError("embedded original-range shape does not match original")
    if original_array.size == 0:
        return 0.0
    return float(np.max(np.abs(embedded - original_array)))
