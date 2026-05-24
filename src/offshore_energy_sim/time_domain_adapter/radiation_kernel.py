"""Radiation-memory kernel construction and diagnostics for the adapter layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from offshore_energy_sim.time_domain.hydrodynamic_memory import (
    apply_radiation_frequency_window,
    project_matrix_series_positive_semidefinite,
    radiation_irf_from_damping,
)


@dataclass(frozen=True)
class RadiationKernelDiagnostics:
    """Scalar stability indicators for a Cummins radiation kernel."""

    time_count: int
    dof_count: int
    zero_lag_frobenius_norm: float
    peak_frobenius_norm: float
    tail_rms_to_peak_ratio: float
    tail_peak_to_peak_ratio: float
    norm_oscillation_score: float
    signed_trace_tail_mean_ratio: float
    memory_integral_frobenius_norm: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def build_radiation_kernel(
    omega: np.ndarray,
    radiation_damping: np.ndarray,
    time: np.ndarray,
    *,
    damping_convention: str = "physical",
    passivity_correction: str = "none",
    frequency_window: str = "none",
    window_start_omega: float | None = None,
    window_stop_omega: float | None = None,
) -> np.ndarray:
    """Build a Cummins/WEC-Sim-style radiation impulse response function."""

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    damping = np.asarray(radiation_damping, dtype=float)
    if passivity_correction == "clip_negative_eigenvalues":
        damping = project_matrix_series_positive_semidefinite(damping)
    elif passivity_correction != "none":
        raise ValueError("passivity_correction must be 'none' or 'clip_negative_eigenvalues'")
    damping = apply_radiation_frequency_window(
        omega_values,
        damping,
        window=frequency_window,
        start_omega=window_start_omega,
        stop_omega=window_stop_omega,
    )
    return radiation_irf_from_damping(
        omega_values,
        damping,
        time,
        damping_convention=damping_convention,
    )


def _kernel_norm(kernel: np.ndarray) -> np.ndarray:
    return np.linalg.norm(kernel.reshape(kernel.shape[0], -1), axis=1)


def _oscillation_score(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    first = np.diff(values)
    active = np.abs(first) > 1.0e-12 * max(float(np.max(np.abs(values))), 1.0)
    if np.count_nonzero(active) < 2:
        return 0.0
    signs = np.sign(first[active])
    return float(np.count_nonzero(signs[1:] * signs[:-1] < 0.0) / max(signs.size - 1, 1))


def radiation_kernel_diagnostics(
    time: np.ndarray,
    radiation_kernel: np.ndarray,
    *,
    tail_fraction: float = 0.25,
) -> RadiationKernelDiagnostics:
    """Return smoothness and decay diagnostics for a radiation-memory kernel."""

    t = np.asarray(time, dtype=float).reshape(-1)
    kernel = np.asarray(radiation_kernel, dtype=float)
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_kernel must have shape (n_time, ndof, ndof)")
    if kernel.shape[0] != t.size:
        raise ValueError("radiation_kernel first axis must match time")
    if t.size < 2:
        raise ValueError("time must contain at least two values")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")

    norms = _kernel_norm(kernel)
    peak = max(float(np.max(norms)), 1.0e-30)
    tail_count = max(1, int(np.ceil(tail_fraction * t.size)))
    tail_norms = norms[-tail_count:]
    trace = np.trace(kernel, axis1=1, axis2=2)
    integral = np.trapz(kernel, t, axis=0)
    return RadiationKernelDiagnostics(
        time_count=int(t.size),
        dof_count=int(kernel.shape[1]),
        zero_lag_frobenius_norm=float(norms[0]),
        peak_frobenius_norm=float(peak),
        tail_rms_to_peak_ratio=float(np.sqrt(np.mean(tail_norms**2)) / peak),
        tail_peak_to_peak_ratio=float(np.max(tail_norms) / peak),
        norm_oscillation_score=_oscillation_score(norms),
        signed_trace_tail_mean_ratio=float(abs(np.mean(trace[-tail_count:])) / max(abs(float(trace[0])), 1.0e-30)),
        memory_integral_frobenius_norm=float(np.linalg.norm(integral)),
    )


def compare_radiation_kernels(
    time: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    tail_fraction: float = 0.25,
) -> dict[str, float | int | dict[str, float | int]]:
    """Compare two radiation kernels on the same time grid."""

    before_diag = radiation_kernel_diagnostics(time, before, tail_fraction=tail_fraction)
    after_diag = radiation_kernel_diagnostics(time, after, tail_fraction=tail_fraction)
    before_norm = _kernel_norm(np.asarray(before, dtype=float))
    after_norm = _kernel_norm(np.asarray(after, dtype=float))
    return {
        "before": before_diag.to_dict(),
        "after": after_diag.to_dict(),
        "tail_rms_ratio_after_over_before": after_diag.tail_rms_to_peak_ratio
        / max(before_diag.tail_rms_to_peak_ratio, 1.0e-30),
        "tail_peak_ratio_after_over_before": after_diag.tail_peak_to_peak_ratio
        / max(before_diag.tail_peak_to_peak_ratio, 1.0e-30),
        "kernel_norm_l2_relative_difference": float(
            np.linalg.norm(after_norm - before_norm) / max(float(np.linalg.norm(before_norm)), 1.0e-30)
        ),
    }
