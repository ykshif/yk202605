"""Adapter-owned state-space radiation solver.

The solver in this module keeps the state-space radiation path outside the
RODM frequency-domain core. It consumes already prepared linear matrices,
forces, and a fitted adapter-layer radiation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offshore_energy_sim.time_domain.solver import LinearTimeDomainResult
from offshore_energy_sim.time_domain_adapter.state_space_radiation import (
    DiscreteStateSpaceRadiationModel,
)


@dataclass(frozen=True)
class StateSpaceRadiationLinearSystem:
    """Linear system with ERA radiation-memory states.

    ``mass``, ``damping``, and ``stiffness`` are the already assembled effective
    matrices for the adapter solve. For a Cummins-style RODM solve this usually
    means ``mass = M_struct + A_inf + A_residual`` and
    ``damping = B_residual + B_other``.
    """

    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    force: np.ndarray
    time: np.ndarray
    radiation_model: DiscreteStateSpaceRadiationModel
    initial_displacement: np.ndarray | None = None
    initial_velocity: np.ndarray | None = None


def _square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def _time_matrix(values: np.ndarray, ndof: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != ndof:
        raise ValueError(f"{name} must have shape (n_time, {ndof})")
    return array


def _interpolate_force(force: np.ndarray, index: int, theta: float) -> np.ndarray:
    return (1.0 - theta) * force[index] + theta * force[index + 1]


def solve_state_space_radiation_linear_system_rk4(
    system: StateSpaceRadiationLinearSystem,
    *,
    implicit_radiation_zero_lag: bool = True,
    radiation_convolution_rule: str = "trapezoidal",
) -> LinearTimeDomainResult:
    """Solve the reduced state-space radiation system with RK4 mechanics.

    The ERA radiation state remains a discrete-time history realization. At
    each grid step it is advanced from known velocity history, and the resulting
    radiation force is linearly interpolated inside the mechanical RK4 step.
    """

    M = _square_matrix(system.mass, "mass")
    C = _square_matrix(system.damping, "damping")
    K = _square_matrix(system.stiffness, "stiffness")
    if M.shape != C.shape or M.shape != K.shape:
        raise ValueError("mass, damping, and stiffness must have the same shape")
    ndof = M.shape[0]
    model = system.radiation_model
    if model.dof_count != ndof:
        raise ValueError("radiation_model DOF count must match mass matrix")

    time = np.asarray(system.time, dtype=float).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError("time must be strictly increasing")
    if not np.isclose(dt, model.time_step):
        raise ValueError("time step must match the state-space radiation model")
    rule = str(radiation_convolution_rule).lower()
    if rule not in {"rectangular", "trapezoidal"}:
        raise ValueError("radiation_convolution_rule must be 'rectangular' or 'trapezoidal'")

    F = _time_matrix(system.force, ndof, "force")
    if F.shape[0] != time.size:
        raise ValueError("force first axis must match time")

    q = np.zeros((time.size, ndof), dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    memory = np.zeros_like(q)
    if system.initial_displacement is not None:
        q[0] = np.asarray(system.initial_displacement, dtype=float).reshape(ndof)
    if system.initial_velocity is not None:
        v[0] = np.asarray(system.initial_velocity, dtype=float).reshape(ndof)

    radiation_zero_lag_damping = np.zeros_like(C)
    if implicit_radiation_zero_lag:
        zero_lag_weight = 0.5 if rule == "trapezoidal" else 1.0
        radiation_zero_lag_damping = zero_lag_weight * dt * model.zero_lag_kernel
    C_step = C + radiation_zero_lag_damping

    radiation_state = np.zeros(model.order, dtype=float)

    def acceleration(
        q_value: np.ndarray,
        v_value: np.ndarray,
        force_value: np.ndarray,
        history_memory: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.solve(M, force_value - history_memory - C_step @ v_value - K @ q_value)

    a[0] = acceleration(q[0], v[0], F[0], np.zeros(ndof, dtype=float))

    for index in range(time.size - 1):
        next_index = index + 1
        history_start = model.output_matrix @ radiation_state
        next_radiation_state = model.state_matrix @ radiation_state + model.input_matrix @ v[index]
        history_end = model.output_matrix @ next_radiation_state

        def derivative(theta: float, q_value: np.ndarray, v_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            force_value = _interpolate_force(F, index, theta)
            history_value = (1.0 - theta) * history_start + theta * history_end
            return v_value, acceleration(q_value, v_value, force_value, history_value)

        k1_q, k1_v = derivative(0.0, q[index], v[index])
        k2_q, k2_v = derivative(0.5, q[index] + 0.5 * dt * k1_q, v[index] + 0.5 * dt * k1_v)
        k3_q, k3_v = derivative(0.5, q[index] + 0.5 * dt * k2_q, v[index] + 0.5 * dt * k2_v)
        k4_q, k4_v = derivative(1.0, q[index] + dt * k3_q, v[index] + dt * k3_v)

        q[next_index] = q[index] + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        v[next_index] = v[index] + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        radiation_state = next_radiation_state
        memory[next_index] = history_end + radiation_zero_lag_damping @ v[next_index]
        a[next_index] = acceleration(q[next_index], v[next_index], F[next_index], history_end)

    return LinearTimeDomainResult(
        time=time,
        displacement=q,
        velocity=v,
        acceleration=a,
        memory_force=memory,
    )


def solve_state_space_radiation_linear_system(
    system: StateSpaceRadiationLinearSystem,
    *,
    implicit_radiation_zero_lag: bool = True,
    radiation_convolution_rule: str = "trapezoidal",
    integrator: str = "newmark",
    newmark_beta: float = 0.25,
    newmark_gamma: float = 0.5,
) -> LinearTimeDomainResult:
    """Solve ``M qdd + C qd + K q + F_rad_state = F(t)``.

    The state update is explicit in the same sense as the current direct
    Cummins-convolution baseline: the radiation-memory force at ``t[n+1]`` uses
    velocity samples known at the start of the time step.
    """

    method = str(integrator).lower()
    if method == "rk4":
        return solve_state_space_radiation_linear_system_rk4(
            system,
            implicit_radiation_zero_lag=implicit_radiation_zero_lag,
            radiation_convolution_rule=radiation_convolution_rule,
        )
    if method != "newmark":
        raise ValueError("integrator must be 'newmark' or 'rk4'")

    M = _square_matrix(system.mass, "mass")
    C = _square_matrix(system.damping, "damping")
    K = _square_matrix(system.stiffness, "stiffness")
    if M.shape != C.shape or M.shape != K.shape:
        raise ValueError("mass, damping, and stiffness must have the same shape")
    ndof = M.shape[0]
    model = system.radiation_model
    if model.dof_count != ndof:
        raise ValueError("radiation_model DOF count must match mass matrix")

    time = np.asarray(system.time, dtype=float).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0]):
        raise ValueError("time must be uniformly spaced")
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError("time must be strictly increasing")
    if not np.isclose(dt, model.time_step):
        raise ValueError("time step must match the state-space radiation model")
    if newmark_beta <= 0.0 or newmark_gamma <= 0.0:
        raise ValueError("Newmark beta and gamma must be positive")

    rule = str(radiation_convolution_rule).lower()
    if rule not in {"rectangular", "trapezoidal"}:
        raise ValueError("radiation_convolution_rule must be 'rectangular' or 'trapezoidal'")
    F = _time_matrix(system.force, ndof, "force")
    if F.shape[0] != time.size:
        raise ValueError("force first axis must match time")

    q = np.zeros((time.size, ndof), dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    memory = np.zeros_like(q)
    if system.initial_displacement is not None:
        q[0] = np.asarray(system.initial_displacement, dtype=float).reshape(ndof)
    if system.initial_velocity is not None:
        v[0] = np.asarray(system.initial_velocity, dtype=float).reshape(ndof)

    radiation_zero_lag_damping = np.zeros_like(C)
    if implicit_radiation_zero_lag:
        zero_lag_weight = 0.5 if rule == "trapezoidal" else 1.0
        radiation_zero_lag_damping = zero_lag_weight * dt * model.zero_lag_kernel
    C_step = C + radiation_zero_lag_damping

    radiation_state = np.zeros(model.order, dtype=float)
    a[0] = np.linalg.solve(M, F[0] - C_step @ v[0] - K @ q[0])
    effective = M + newmark_gamma * dt * C_step + newmark_beta * dt**2 * K

    for index in range(time.size - 1):
        next_index = index + 1
        radiation_state = model.state_matrix @ radiation_state + model.input_matrix @ v[index]
        history_memory = model.output_matrix @ radiation_state

        q_predict = q[index] + dt * v[index] + dt**2 * (0.5 - newmark_beta) * a[index]
        v_predict = v[index] + dt * (1.0 - newmark_gamma) * a[index]
        rhs = F[next_index] - history_memory - C_step @ v_predict - K @ q_predict
        a[next_index] = np.linalg.solve(effective, rhs)
        q[next_index] = q_predict + newmark_beta * dt**2 * a[next_index]
        v[next_index] = v_predict + newmark_gamma * dt * a[next_index]
        memory[next_index] = history_memory + radiation_zero_lag_damping @ v[next_index]

    return LinearTimeDomainResult(
        time=time,
        displacement=q,
        velocity=v,
        acceleration=a,
        memory_force=memory,
    )
