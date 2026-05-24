"""State-space radiation-memory approximation for the adapter layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StateSpaceRadiationModel:
    """Common-pole exponential approximation of a radiation memory kernel.

    The model represents a matrix-valued Cummins kernel as

    ``K(t) ~= sum_i R_i exp(-p_i t)``

    with positive real poles ``p_i`` and matrix residues ``R_i``. This gives a
    compact MIMO state-space realization with one state vector per pole:

    ``zdot_i = -p_i z_i + qdot`` and ``F_rad = sum_i R_i z_i``.
    """

    poles: np.ndarray
    residues: np.ndarray
    fit_l2_relative_error: float
    fit_peak_relative_error: float

    def __post_init__(self) -> None:
        poles = np.asarray(self.poles, dtype=float).reshape(-1)
        residues = np.asarray(self.residues, dtype=float)
        if poles.size == 0:
            raise ValueError("poles must contain at least one value")
        if np.any(poles <= 0.0):
            raise ValueError("state-space poles must be positive")
        if residues.ndim != 3 or residues.shape[0] != poles.size or residues.shape[1] != residues.shape[2]:
            raise ValueError("residues must have shape (n_poles, ndof, ndof)")
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)

    @property
    def order(self) -> int:
        return int(self.poles.size)

    @property
    def dof_count(self) -> int:
        return int(self.residues.shape[1])

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["poles"] = self.poles.tolist()
        data["residue_shape"] = list(self.residues.shape)
        data.pop("residues", None)
        return data


@dataclass(frozen=True)
class DiscreteStateSpaceRadiationModel:
    """Discrete MIMO state-space approximation of radiation-memory history.

    The model approximates the explicit convolution weights
    ``G_k = dt * K(k dt), k >= 1`` with the Markov sequence
    ``G_k ~= C A^(k-1) B``. The zero-lag kernel is stored separately so a
    solver can keep the same implicit zero-lag treatment as the direct
    Cummins-convolution baseline.
    """

    state_matrix: np.ndarray
    input_matrix: np.ndarray
    output_matrix: np.ndarray
    time_step: float
    zero_lag_kernel: np.ndarray
    fit_l2_relative_error: float
    spectral_radius: float

    def __post_init__(self) -> None:
        a = np.asarray(self.state_matrix, dtype=float)
        b = np.asarray(self.input_matrix, dtype=float)
        c = np.asarray(self.output_matrix, dtype=float)
        k0 = np.asarray(self.zero_lag_kernel, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("state_matrix must be square")
        if b.ndim != 2 or b.shape[0] != a.shape[0]:
            raise ValueError("input_matrix must have shape (n_state, ndof)")
        if c.ndim != 2 or c.shape[1] != a.shape[0]:
            raise ValueError("output_matrix must have shape (ndof, n_state)")
        if k0.shape != (c.shape[0], b.shape[1]):
            raise ValueError("zero_lag_kernel shape must match output/input DOFs")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        object.__setattr__(self, "state_matrix", a)
        object.__setattr__(self, "input_matrix", b)
        object.__setattr__(self, "output_matrix", c)
        object.__setattr__(self, "zero_lag_kernel", k0)

    @property
    def order(self) -> int:
        return int(self.state_matrix.shape[0])

    @property
    def dof_count(self) -> int:
        return int(self.input_matrix.shape[1])

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "dof_count": self.dof_count,
            "time_step": float(self.time_step),
            "fit_l2_relative_error": float(self.fit_l2_relative_error),
            "spectral_radius": float(self.spectral_radius),
            "state_matrix_shape": list(self.state_matrix.shape),
            "input_matrix_shape": list(self.input_matrix.shape),
            "output_matrix_shape": list(self.output_matrix.shape),
            "zero_lag_kernel_shape": list(self.zero_lag_kernel.shape),
        }


def default_real_poles(
    time: np.ndarray,
    *,
    order: int,
    slowest_time_scale: float | None = None,
    fastest_time_scale: float | None = None,
) -> np.ndarray:
    """Return logarithmically spaced stable real poles for IRF fitting."""

    if order < 1:
        raise ValueError("order must be positive")
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("time must contain at least two values")
    steps = np.diff(t)
    if np.any(steps <= 0.0):
        raise ValueError("time must be strictly increasing")
    dt = float(np.median(steps))
    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError("time duration must be positive")
    slow = duration if slowest_time_scale is None else float(slowest_time_scale)
    fast = max(2.0 * dt, duration / 200.0) if fastest_time_scale is None else float(fastest_time_scale)
    if slow <= 0.0 or fast <= 0.0:
        raise ValueError("time scales must be positive")
    if fast > slow:
        raise ValueError("fastest_time_scale must be less than or equal to slowest_time_scale")
    time_scales = np.geomspace(slow, fast, int(order))
    return 1.0 / time_scales


def fit_common_pole_state_space_radiation(
    time: np.ndarray,
    radiation_kernel: np.ndarray,
    *,
    order: int = 8,
    poles: np.ndarray | None = None,
    enforce_symmetric_residues: bool = True,
    ridge_alpha: float = 0.0,
    rcond: float | None = None,
) -> StateSpaceRadiationModel:
    """Fit a common-pole state-space approximation to a matrix IRF."""

    t = np.asarray(time, dtype=float).reshape(-1)
    kernel = np.asarray(radiation_kernel, dtype=float)
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_kernel must have shape (n_time, ndof, ndof)")
    if kernel.shape[0] != t.size:
        raise ValueError("radiation_kernel first axis must match time")
    pole_values = default_real_poles(t, order=order) if poles is None else np.asarray(poles, dtype=float).reshape(-1)
    if pole_values.size != int(order) and poles is None:
        raise RuntimeError("default pole generation returned an unexpected order")
    if np.any(pole_values <= 0.0):
        raise ValueError("poles must be positive")
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")

    basis = np.exp(-np.outer(t - t[0], pole_values))
    flat_kernel = kernel.reshape(t.size, -1)
    if ridge_alpha > 0.0:
        normal = basis.T @ basis
        rhs = basis.T @ flat_kernel
        scale = max(float(np.trace(normal) / normal.shape[0]), 1.0e-30)
        coefficients = np.linalg.solve(
            normal + float(ridge_alpha) * scale * np.eye(normal.shape[0]),
            rhs,
        )
    else:
        coefficients, *_ = np.linalg.lstsq(basis, flat_kernel, rcond=rcond)
    residues = coefficients.reshape(pole_values.size, kernel.shape[1], kernel.shape[2])
    if enforce_symmetric_residues:
        residues = 0.5 * (residues + np.swapaxes(residues, 1, 2))

    fitted = evaluate_state_space_radiation_kernel(
        StateSpaceRadiationModel(
            poles=pole_values,
            residues=residues,
            fit_l2_relative_error=0.0,
            fit_peak_relative_error=0.0,
        ),
        t,
    )
    residual = fitted - kernel
    kernel_norm = max(float(np.linalg.norm(kernel)), 1.0e-30)
    peak_norm = max(float(np.max(np.linalg.norm(kernel.reshape(t.size, -1), axis=1))), 1.0e-30)
    peak_residual = float(np.max(np.linalg.norm(residual.reshape(t.size, -1), axis=1)))
    return StateSpaceRadiationModel(
        poles=pole_values,
        residues=residues,
        fit_l2_relative_error=float(np.linalg.norm(residual) / kernel_norm),
        fit_peak_relative_error=float(peak_residual / peak_norm),
    )


def evaluate_state_space_radiation_kernel(
    model: StateSpaceRadiationModel,
    time: np.ndarray,
) -> np.ndarray:
    """Evaluate ``K(t)`` from a fitted common-pole radiation model."""

    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("time must contain at least one value")
    basis = np.exp(-np.outer(t - t[0], model.poles))
    return np.einsum("tp,pij->tij", basis, model.residues)


def state_space_radiation_coefficients(
    model: StateSpaceRadiationModel,
    omega: float | np.ndarray,
    *,
    added_mass_infinite: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return frequency-domain ``A(omega), B(omega)`` from the state model."""

    omega_values = np.asarray(omega, dtype=float).reshape(-1)
    if np.any(omega_values < 0.0):
        raise ValueError("omega must be non-negative")
    poles = model.poles
    denom = poles[:, np.newaxis] ** 2 + omega_values[np.newaxis, :] ** 2
    damping_weights = poles[:, np.newaxis] / denom
    damping = np.einsum("pf,pij->fij", damping_weights, model.residues)

    added_mass = None
    if added_mass_infinite is not None:
        a_inf = np.asarray(added_mass_infinite, dtype=float)
        if a_inf.shape != model.residues.shape[1:]:
            raise ValueError("added_mass_infinite shape must match model matrix shape")
        mass_weights = 1.0 / denom
        added_mass = a_inf[np.newaxis, :, :] - np.einsum("pf,pij->fij", mass_weights, model.residues)

    if np.ndim(omega) == 0:
        return (
            None if added_mass is None else added_mass[0],
            damping[0],
        )
    return added_mass, damping


def simulate_state_space_memory_force(
    velocity: np.ndarray,
    time: np.ndarray,
    model: StateSpaceRadiationModel,
) -> np.ndarray:
    """Return radiation-memory force from the fitted state-space model.

    The update uses exact zero-order-hold integration over each time step using
    the previous velocity sample. This mirrors the explicit history treatment
    used by the current direct-convolution baseline.
    """

    velocity_values = np.asarray(velocity, dtype=float)
    t = np.asarray(time, dtype=float).reshape(-1)
    if velocity_values.ndim != 2 or velocity_values.shape[1] != model.dof_count:
        raise ValueError("velocity must have shape (n_time, model.dof_count)")
    if velocity_values.shape[0] != t.size:
        raise ValueError("velocity first axis must match time")
    if t.size < 2:
        raise ValueError("time must contain at least two values")
    steps = np.diff(t)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    dt = float(steps[0])

    states = np.zeros((model.order, model.dof_count), dtype=float)
    force = np.zeros_like(velocity_values)
    alpha = np.exp(-model.poles * dt)
    beta = (1.0 - alpha) / model.poles
    for index in range(1, t.size):
        states = alpha[:, np.newaxis] * states + beta[:, np.newaxis] * velocity_values[index - 1]
        force[index] = np.einsum("pij,pj->i", model.residues, states)
    return force


def relative_l2_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Return a safe relative L2 error."""

    target = np.asarray(reference, dtype=float)
    residual = np.asarray(predicted, dtype=float) - target
    return float(np.linalg.norm(residual) / max(float(np.linalg.norm(target)), 1.0e-30))


def _block_hankel(markov: np.ndarray, block_rows: int, block_cols: int, *, offset: int) -> np.ndarray:
    output_count, input_count = markov.shape[1:]
    hankel = np.empty((block_rows * output_count, block_cols * input_count), dtype=float)
    for row in range(block_rows):
        for column in range(block_cols):
            block = markov[row + column + offset]
            row_slice = slice(row * output_count, (row + 1) * output_count)
            column_slice = slice(column * input_count, (column + 1) * input_count)
            hankel[row_slice, column_slice] = block
    return hankel


def _stabilize_discrete_matrix(matrix: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        raise ValueError("stability radius must be positive")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    clipped = eigenvalues.copy()
    magnitudes = np.abs(clipped)
    unstable = magnitudes >= radius
    clipped[unstable] *= radius / np.maximum(magnitudes[unstable], 1.0e-30)
    stabilized = eigenvectors @ np.diag(clipped) @ np.linalg.inv(eigenvectors)
    return np.real_if_close(stabilized, tol=1000).real


def fit_era_state_space_radiation(
    time: np.ndarray,
    radiation_kernel: np.ndarray,
    *,
    order: int = 40,
    block_rows: int | None = None,
    block_cols: int | None = None,
    stabilize: bool = True,
    stability_radius: float = 0.999,
) -> DiscreteStateSpaceRadiationModel:
    """Fit a discrete state-space radiation model with ERA."""

    t = np.asarray(time, dtype=float).reshape(-1)
    kernel = np.asarray(radiation_kernel, dtype=float)
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError("radiation_kernel must have shape (n_time, ndof, ndof)")
    if kernel.shape[0] != t.size:
        raise ValueError("radiation_kernel first axis must match time")
    if t.size < 4:
        raise ValueError("time must contain at least four samples")
    steps = np.diff(t)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    if order < 1:
        raise ValueError("order must be positive")
    dt = float(steps[0])
    markov = dt * kernel[1:]
    markov_count = markov.shape[0]
    if markov_count < 3:
        raise ValueError("radiation kernel must provide at least three nonzero lags")
    rows = max(2, min(block_rows or 12, (markov_count - 1) // 2))
    cols = max(2, min(block_cols or rows, markov_count - rows))
    if rows + cols > markov_count:
        raise ValueError("ERA block_rows + block_cols must not exceed available Markov lags")

    hankel_0 = _block_hankel(markov, rows, cols, offset=0)
    hankel_1 = _block_hankel(markov, rows, cols, offset=1)
    u, singular_values, vh = np.linalg.svd(hankel_0, full_matrices=False)
    retained = min(int(order), singular_values.size)
    positive = singular_values[:retained] > 1.0e-14 * max(float(singular_values[0]), 1.0)
    retained = max(1, int(np.count_nonzero(positive)))
    u_r = u[:, :retained]
    s_r = singular_values[:retained]
    vh_r = vh[:retained, :]
    sqrt_s = np.diag(np.sqrt(s_r))
    inv_sqrt_s = np.diag(1.0 / np.sqrt(s_r))

    state_matrix = inv_sqrt_s @ u_r.T @ hankel_1 @ vh_r.T @ inv_sqrt_s
    if stabilize:
        state_matrix = _stabilize_discrete_matrix(state_matrix, stability_radius)
    input_count = kernel.shape[2]
    output_count = kernel.shape[1]
    input_matrix = sqrt_s @ vh_r[:, :input_count]
    output_matrix = u_r[:output_count, :] @ sqrt_s
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(state_matrix))))
    model = DiscreteStateSpaceRadiationModel(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        time_step=dt,
        zero_lag_kernel=kernel[0],
        fit_l2_relative_error=0.0,
        spectral_radius=spectral_radius,
    )
    fitted = evaluate_era_markov_parameters(model, markov_count)
    fit_error = relative_l2_error(fitted, markov)
    return DiscreteStateSpaceRadiationModel(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        time_step=dt,
        zero_lag_kernel=kernel[0],
        fit_l2_relative_error=fit_error,
        spectral_radius=spectral_radius,
    )


def evaluate_era_markov_parameters(
    model: DiscreteStateSpaceRadiationModel,
    lag_count: int,
) -> np.ndarray:
    """Return Markov weights ``G_k = C A^(k-1) B`` for ``k = 1..lag_count``."""

    if lag_count < 1:
        raise ValueError("lag_count must be positive")
    markov = np.empty((int(lag_count), model.dof_count, model.dof_count), dtype=float)
    state_power_input = model.input_matrix.copy()
    for index in range(int(lag_count)):
        markov[index] = model.output_matrix @ state_power_input
        state_power_input = model.state_matrix @ state_power_input
    return markov


def evaluate_era_radiation_kernel(
    model: DiscreteStateSpaceRadiationModel,
    lag_count: int,
) -> np.ndarray:
    """Evaluate an equivalent sampled kernel from an ERA model."""

    if lag_count < 2:
        raise ValueError("lag_count must be at least two")
    kernel = np.empty((int(lag_count), model.dof_count, model.dof_count), dtype=float)
    kernel[0] = model.zero_lag_kernel
    kernel[1:] = evaluate_era_markov_parameters(model, int(lag_count) - 1) / model.time_step
    return kernel


def simulate_era_memory_force(
    velocity: np.ndarray,
    time: np.ndarray,
    model: DiscreteStateSpaceRadiationModel,
) -> np.ndarray:
    """Return explicit radiation-memory force from a discrete ERA model."""

    velocity_values = np.asarray(velocity, dtype=float)
    t = np.asarray(time, dtype=float).reshape(-1)
    if velocity_values.ndim != 2 or velocity_values.shape[1] != model.dof_count:
        raise ValueError("velocity must have shape (n_time, model.dof_count)")
    if velocity_values.shape[0] != t.size:
        raise ValueError("velocity first axis must match time")
    if t.size < 2:
        raise ValueError("time must contain at least two values")
    steps = np.diff(t)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    if not np.isclose(float(steps[0]), model.time_step):
        raise ValueError("velocity time step must match the ERA model time_step")

    state = np.zeros(model.order, dtype=float)
    force = np.zeros_like(velocity_values)
    for index in range(1, t.size):
        state = model.state_matrix @ state + model.input_matrix @ velocity_values[index - 1]
        force[index] = model.output_matrix @ state
    return force


def save_discrete_state_space_radiation_model(
    model: DiscreteStateSpaceRadiationModel,
    path: str | Path,
) -> Path:
    """Save an ERA radiation model to a compressed NumPy archive."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state_matrix=model.state_matrix,
        input_matrix=model.input_matrix,
        output_matrix=model.output_matrix,
        zero_lag_kernel=model.zero_lag_kernel,
        time_step=np.array(model.time_step, dtype=float),
        fit_l2_relative_error=np.array(model.fit_l2_relative_error, dtype=float),
        spectral_radius=np.array(model.spectral_radius, dtype=float),
    )
    return output


def load_discrete_state_space_radiation_model(
    path: str | Path,
) -> DiscreteStateSpaceRadiationModel:
    """Load an ERA radiation model saved by ``save_discrete_state_space_radiation_model``."""

    with np.load(Path(path), allow_pickle=False) as archive:
        return DiscreteStateSpaceRadiationModel(
            state_matrix=archive["state_matrix"],
            input_matrix=archive["input_matrix"],
            output_matrix=archive["output_matrix"],
            zero_lag_kernel=archive["zero_lag_kernel"],
            time_step=float(archive["time_step"]),
            fit_l2_relative_error=float(archive["fit_l2_relative_error"]),
            spectral_radius=float(archive["spectral_radius"]),
        )
