"""Hydrodynamic radiation-memory preprocessing for Cummins equations."""

from __future__ import annotations

import numpy as np


def project_symmetric_positive_semidefinite(
    matrix: np.ndarray,
    *,
    minimum_eigenvalue: float = 0.0,
) -> np.ndarray:
    """Return the nearest eigenvalue-clipped symmetric PSD matrix."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, minimum_eigenvalue)
    return (eigenvectors * clipped[np.newaxis, :]) @ eigenvectors.T


def project_matrix_series_positive_semidefinite(
    matrices: np.ndarray,
    *,
    minimum_eigenvalue: float = 0.0,
) -> np.ndarray:
    """Apply symmetric PSD projection to a frequency-major matrix series."""

    series = np.asarray(matrices, dtype=float)
    if series.ndim != 3 or series.shape[1] != series.shape[2]:
        raise ValueError("matrices must have shape (n, ndof, ndof)")
    return np.stack(
        [
            project_symmetric_positive_semidefinite(
                matrix,
                minimum_eigenvalue=minimum_eigenvalue,
            )
            for matrix in series
        ],
        axis=0,
    )


def radiation_frequency_window_weights(
    omega: np.ndarray,
    *,
    window: str = "none",
    start_omega: float | None = None,
    stop_omega: float | None = None,
) -> np.ndarray:
    """Return finite-band weights for radiation damping before IRF generation.

    The weights are intended for exploratory Cummins preprocessing. They leave
    low-frequency data unchanged and taper the high-frequency tail to reduce
    finite-band ringing in the cosine transform.
    """

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    if omega_values.size == 0:
        raise ValueError("omega must contain at least one value")
    if np.any(np.diff(omega_values) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    window_name = str(window).lower()
    if window_name == "none":
        return np.ones_like(omega_values)
    if window_name not in {"linear_tail", "cosine_tail"}:
        raise ValueError("unsupported radiation frequency window")

    omega_min = float(omega_values[0])
    omega_max = float(omega_values[-1])
    if start_omega is None:
        start = omega_min + 0.75 * (omega_max - omega_min)
    else:
        start = float(start_omega)
    stop = omega_max if stop_omega is None else float(stop_omega)
    if stop <= start:
        raise ValueError("radiation window stop_omega must be greater than start_omega")

    weights = np.ones_like(omega_values)
    tail = omega_values > start
    weights[omega_values >= stop] = 0.0
    transition = tail & (omega_values < stop)
    xi = (omega_values[transition] - start) / (stop - start)
    if window_name == "linear_tail":
        weights[transition] = 1.0 - xi
    else:
        weights[transition] = 0.5 * (1.0 + np.cos(np.pi * xi))
    return weights


def apply_radiation_frequency_window(
    omega: np.ndarray,
    radiation_damping: np.ndarray,
    *,
    window: str = "none",
    start_omega: float | None = None,
    stop_omega: float | None = None,
) -> np.ndarray:
    """Apply a scalar frequency window to a radiation-damping matrix series."""

    damping = np.asarray(radiation_damping, dtype=float)
    weights = radiation_frequency_window_weights(
        omega,
        window=window,
        start_omega=start_omega,
        stop_omega=stop_omega,
    )
    if damping.shape[0] != weights.size:
        raise ValueError("radiation_damping first axis must match omega")
    return weights[:, np.newaxis, np.newaxis] * damping


def radiation_irf_from_damping(
    omega: np.ndarray,
    radiation_damping: np.ndarray,
    time: np.ndarray,
    *,
    damping_convention: str = "physical",
) -> np.ndarray:
    """Build the radiation impulse-response function from damping matrices.

    For physical Capytaine-style damping matrices this uses the Cummins/WEC-Sim
    cosine-transform convention
    ``K(t) = 2/pi * integral_0^infty B(omega) cos(omega*t) d omega``.

    ``damping_convention='wec_sim_bemio'`` is provided for WEC-Sim/BEMIO-style
    normalized damping values where the integrand is proportional to
    ``omega * Bbar(omega)``.
    """

    omega = np.asarray(omega, dtype=float).reshape(-1)
    damping = np.asarray(radiation_damping, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if damping.shape[0] != omega.size:
        raise ValueError("radiation_damping first axis must match omega")
    if damping.ndim != 3 or damping.shape[1] != damping.shape[2]:
        raise ValueError("radiation_damping must have shape (n_omega, ndof, ndof)")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    if damping_convention == "physical":
        integrand = damping
    elif damping_convention == "wec_sim_bemio":
        integrand = omega[:, np.newaxis, np.newaxis] * damping
    else:
        raise ValueError("unsupported damping_convention")

    cos_terms = np.cos(np.outer(time, omega))
    # IRF shape: (n_time, ndof, ndof). Each entry integrates along omega.
    return (2.0 / np.pi) * np.trapz(
        cos_terms[:, :, np.newaxis, np.newaxis] * integrand[np.newaxis, :, :, :],
        omega,
        axis=1,
    )


def estimate_infinite_frequency_added_mass(
    omega: np.ndarray,
    added_mass: np.ndarray,
    *,
    method: str = "high_frequency",
    tail_count: int = 3,
) -> np.ndarray:
    """Estimate the infinite-frequency added-mass matrix.

    The safest first implementation for this repository is a documented
    high-frequency limit estimate. It averages the last ``tail_count`` added
    mass matrices from the available BEM frequency grid.
    """

    omega = np.asarray(omega, dtype=float).reshape(-1)
    mass = np.asarray(added_mass, dtype=float)
    if mass.shape[0] != omega.size:
        raise ValueError("added_mass first axis must match omega")
    if mass.ndim != 3 or mass.shape[1] != mass.shape[2]:
        raise ValueError("added_mass must have shape (n_omega, ndof, ndof)")
    if method != "high_frequency":
        raise ValueError("only method='high_frequency' is supported by this function")
    if tail_count < 1:
        raise ValueError("tail_count must be positive")
    count = min(int(tail_count), omega.size)
    return np.mean(mass[-count:, :, :], axis=0)


def radiation_coefficients_from_irf(
    omega: float | np.ndarray,
    radiation_irf: np.ndarray,
    time: np.ndarray,
    *,
    added_mass_infinite: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Recover frequency-domain radiation coefficients from an IRF.

    For the Cummins convention used by WEC-Sim:

    ``B(omega) = integral_0^infty K(t) cos(omega t) dt``

    and, when ``A_inf`` is supplied,

    ``A(omega) = A_inf - 1/omega * integral_0^infty K(t) sin(omega t) dt``.

    The returned arrays are frequency-major when ``omega`` contains multiple
    values. For scalar ``omega`` the leading frequency axis is squeezed.
    """

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    kernel = np.asarray(radiation_irf, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_irf must have shape (n_time, ndof, ndof)")
    if kernel.shape[0] != time.size:
        raise ValueError("radiation_irf first axis must match time")
    if np.any(omega_values < 0.0):
        raise ValueError("omega must be non-negative")

    cos_terms = np.cos(np.outer(omega_values, time))
    damping = np.trapz(
        cos_terms[:, :, np.newaxis, np.newaxis] * kernel[np.newaxis, :, :, :],
        time,
        axis=1,
    )

    added_mass = None
    if added_mass_infinite is not None:
        a_inf = np.asarray(added_mass_infinite, dtype=float)
        if a_inf.shape != kernel.shape[1:]:
            raise ValueError("added_mass_infinite shape must match IRF matrix shape")
        added_mass = np.empty_like(damping)
        sin_terms = np.sin(np.outer(omega_values, time))
        sin_integral = np.trapz(
            sin_terms[:, :, np.newaxis, np.newaxis] * kernel[np.newaxis, :, :, :],
            time,
            axis=1,
        )
        for index, omega_value in enumerate(omega_values):
            if omega_value <= 0.0:
                added_mass[index] = a_inf
            else:
                added_mass[index] = a_inf - sin_integral[index] / omega_value

    if np.ndim(omega) == 0:
        return (
            None if added_mass is None else added_mass[0],
            damping[0],
        )
    return added_mass, damping


def radiation_coefficients_from_discrete_irf(
    omega: float | np.ndarray,
    radiation_irf: np.ndarray,
    time: np.ndarray,
    *,
    added_mass_infinite: np.ndarray | None = None,
    convolution_rule: str = "rectangular",
) -> tuple[np.ndarray | None, np.ndarray]:
    """Recover radiation coefficients for the solver's rectangular convolution.

    ``solve_linear_time_domain`` applies the radiation-memory kernel with a
    full-weight rectangular sum and folds the zero-lag term into the damping
    matrix. This helper mirrors that discrete transfer function, which is the
    right coefficient estimate for selected-frequency residual correction.
    """

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    kernel = np.asarray(radiation_irf, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_irf must have shape (n_time, ndof, ndof)")
    if kernel.shape[0] != time.size:
        raise ValueError("radiation_irf first axis must match time")
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    if np.any(omega_values < 0.0):
        raise ValueError("omega must be non-negative")
    dt = float(steps[0])
    rule = str(convolution_rule).lower()
    if rule not in {"rectangular", "trapezoidal"}:
        raise ValueError("convolution_rule must be 'rectangular' or 'trapezoidal'")
    weights = np.ones(time.size, dtype=float)
    if rule == "trapezoidal":
        weights[0] = 0.5
        weights[-1] = 0.5

    cos_terms = np.cos(np.outer(omega_values, time))
    damping = dt * np.sum(
        weights[np.newaxis, :, np.newaxis, np.newaxis]
        * cos_terms[:, :, np.newaxis, np.newaxis]
        * kernel[np.newaxis, :, :, :],
        axis=1,
    )

    added_mass = None
    if added_mass_infinite is not None:
        a_inf = np.asarray(added_mass_infinite, dtype=float)
        if a_inf.shape != kernel.shape[1:]:
            raise ValueError("added_mass_infinite shape must match IRF matrix shape")
        sin_terms = np.sin(np.outer(omega_values, time))
        sin_integral = dt * np.sum(
            weights[np.newaxis, :, np.newaxis, np.newaxis]
            * sin_terms[:, :, np.newaxis, np.newaxis]
            * kernel[np.newaxis, :, :, :],
            axis=1,
        )
        added_mass = np.empty_like(damping)
        for index, omega_value in enumerate(omega_values):
            if omega_value <= 0.0:
                added_mass[index] = a_inf
            else:
                added_mass[index] = a_inf - sin_integral[index] / omega_value

    if np.ndim(omega) == 0:
        return (
            None if added_mass is None else added_mass[0],
            damping[0],
        )
    return added_mass, damping


def estimate_infinite_frequency_added_mass_from_irf(
    omega: np.ndarray,
    added_mass: np.ndarray,
    radiation_irf: np.ndarray,
    time: np.ndarray,
    *,
    tail_count: int | None = None,
) -> np.ndarray:
    """Estimate ``A_inf`` from Ogilvie's Cummins relation.

    This uses

    ``A_inf = A(omega) + 1/omega * integral_0^infty K(t) sin(omega t) dt``

    and averages the estimates over all positive frequencies, or over the last
    ``tail_count`` positive frequencies when requested.
    """

    omega = np.asarray(omega, dtype=float).reshape(-1)
    mass = np.asarray(added_mass, dtype=float)
    kernel = np.asarray(radiation_irf, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if mass.shape[0] != omega.size:
        raise ValueError("added_mass first axis must match omega")
    if mass.ndim != 3 or mass.shape[1] != mass.shape[2]:
        raise ValueError("added_mass must have shape (n_omega, ndof, ndof)")
    if kernel.ndim != 3 or kernel.shape[1:] != mass.shape[1:]:
        raise ValueError("radiation_irf matrix shape must match added_mass")
    if kernel.shape[0] != time.size:
        raise ValueError("radiation_irf first axis must match time")

    positive = np.flatnonzero(omega > 0.0)
    if positive.size == 0:
        raise ValueError("at least one positive omega is required")
    if tail_count is not None:
        if tail_count < 1:
            raise ValueError("tail_count must be positive")
        positive = positive[-min(int(tail_count), positive.size) :]

    omega_positive = omega[positive]
    sin_terms = np.sin(np.outer(omega_positive, time))
    sin_integral = np.trapz(
        sin_terms[:, :, np.newaxis, np.newaxis] * kernel[np.newaxis, :, :, :],
        time,
        axis=1,
    )
    estimates = mass[positive] + sin_integral / omega_positive[:, np.newaxis, np.newaxis]
    return np.mean(estimates, axis=0)
