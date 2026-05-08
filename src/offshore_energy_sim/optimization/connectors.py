"""Connector-design parameter objects for future optimization studies.

The current module intentionally does not choose an optimization algorithm.
It provides stable data structures so hinge position/stiffness studies can be
described consistently before connecting scipy, genetic algorithms, or custom
parameter sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConnectorVariableKind = Literal["stiffness", "released_dof_stiffness", "enabled"]
ConnectorObjectiveKind = Literal[
    "max_heave",
    "mean_heave",
    "max_connector_force",
    "mean_connector_force",
    "custom",
]


@dataclass(frozen=True)
class ConnectorDesignVariable:
    """One bounded design variable attached to a connector group."""

    name: str
    connector_group: str
    kind: ConnectorVariableKind
    initial_value: float
    lower_bound: float
    upper_bound: float
    scale: float = 1.0

    def normalized_initial_value(self) -> float:
        """Return the initial value mapped to approximately order-one scale."""

        return self.initial_value / self.scale


@dataclass(frozen=True)
class ConnectorObjectiveSpec:
    """Optimization objective placeholder for response/connecter studies."""

    kind: ConnectorObjectiveKind
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class ConnectorOptimizationProblem:
    """Container describing a future connector optimization problem."""

    case_id: str
    design_variables: tuple[ConnectorDesignVariable, ...]
    objectives: tuple[ConnectorObjectiveSpec, ...]
    constraints: tuple[str, ...] = ()


def uniform_hinge_stiffness_variables(
    *,
    initial_k: float = 1.0e10,
    lower_bound: float = 1.0e6,
    upper_bound: float = 1.0e12,
) -> tuple[ConnectorDesignVariable, ...]:
    """Return x/y hinge stiffness variables for modular-grid studies."""

    return (
        ConnectorDesignVariable(
            name="k_hinge_x",
            connector_group="x hinge lines",
            kind="stiffness",
            initial_value=initial_k,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            scale=initial_k,
        ),
        ConnectorDesignVariable(
            name="k_hinge_y",
            connector_group="y hinge lines",
            kind="stiffness",
            initial_value=initial_k,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            scale=initial_k,
        ),
    )
