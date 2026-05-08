"""Small Pareto and constraint helpers for design scans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricObjective:
    """One scalar objective read from an evaluation summary row."""

    name: str
    metric: str
    minimize: bool = True


@dataclass(frozen=True)
class MetricConstraint:
    """One scalar bound constraint read from an evaluation summary row.

    A positive margin means the constraint is satisfied. For an upper bound the
    margin is ``upper_bound - value``; for a lower bound it is
    ``value - lower_bound``.
    """

    name: str
    metric: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("MetricConstraint needs a lower_bound or upper_bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound cannot exceed upper_bound")


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    """Read one numeric metric from a row."""

    if metric not in row:
        raise KeyError(f"Metric {metric!r} is missing from row")
    return float(row[metric])


def objective_matrix(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[MetricObjective],
) -> np.ndarray:
    """Return an ``(n_designs, n_objectives)`` matrix in minimization form."""

    if not objectives:
        raise ValueError("At least one objective is required")

    values = np.empty((len(rows), len(objectives)), dtype=float)
    for row_index, row in enumerate(rows):
        for objective_index, objective in enumerate(objectives):
            value = _metric_value(row, objective.metric)
            values[row_index, objective_index] = value if objective.minimize else -value
    return values


def pareto_mask_from_values(values: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated rows for minimization objectives."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (n_designs, n_objectives)")

    design_count = matrix.shape[0]
    mask = np.ones(design_count, dtype=bool)
    for i in range(design_count):
        if not mask[i]:
            continue
        for j in range(design_count):
            if i == j:
                continue
            no_worse = np.all(matrix[j] <= matrix[i])
            strictly_better = np.any(matrix[j] < matrix[i])
            if no_worse and strictly_better:
                mask[i] = False
                break
    return mask


def constraint_margins(
    row: Mapping[str, Any],
    constraints: Sequence[MetricConstraint],
) -> dict[str, float]:
    """Return one margin per constraint."""

    margins: dict[str, float] = {}
    for constraint in constraints:
        value = _metric_value(row, constraint.metric)
        row_margins: list[float] = []
        if constraint.lower_bound is not None:
            row_margins.append(value - float(constraint.lower_bound))
        if constraint.upper_bound is not None:
            row_margins.append(float(constraint.upper_bound) - value)
        margins[constraint.name] = min(row_margins)
    return margins


def constraints_satisfied(
    row: Mapping[str, Any],
    constraints: Sequence[MetricConstraint],
) -> bool:
    """Return whether all constraints are satisfied."""

    return all(margin >= 0.0 for margin in constraint_margins(row, constraints).values())


def mark_pareto_rows(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[MetricObjective],
    constraints: Sequence[MetricConstraint] = (),
    *,
    feasible_only: bool = True,
) -> list[dict[str, Any]]:
    """Return rows annotated with feasibility and Pareto flags."""

    annotated = [dict(row) for row in rows]
    feasible = np.array(
        [constraints_satisfied(row, constraints) for row in annotated],
        dtype=bool,
    )
    for row, is_feasible in zip(annotated, feasible):
        row["is_feasible"] = bool(is_feasible)
        margins = constraint_margins(row, constraints)
        for name, margin in margins.items():
            row[f"{name}_margin"] = float(margin)

    candidate_indices = np.flatnonzero(feasible) if feasible_only else np.arange(len(annotated))
    pareto = np.zeros(len(annotated), dtype=bool)
    if candidate_indices.size:
        values = objective_matrix([annotated[index] for index in candidate_indices], objectives)
        pareto[candidate_indices] = pareto_mask_from_values(values)

    for row, is_pareto in zip(annotated, pareto):
        row["is_pareto"] = bool(is_pareto)
    return annotated
