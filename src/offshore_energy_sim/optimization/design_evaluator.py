"""Single-frequency design evaluation helpers.

This module keeps the first optimization step deliberately small: evaluate one
design at one wave frequency, then report response and connector-force metrics.
It does not choose an optimization algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from offshore_energy_sim.strength import (
    build_case_hinge_pair_connectors,
    harmonic_vector_norm_envelope,
    recover_connector_response,
)
from offshore_energy_sim.optimization.hinge_design_space import (
    HingeGrouping,
    HingeParameter,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (
    build_complex_hinge_10x10_case,
    extract_complex_hinge_heave_grid,
    solve_complex_hinge_case,
)


@dataclass(frozen=True)
class SingleFrequencyScenario:
    """One external load scenario for fixed-frequency evaluation.

    Parameters
    ----------
    omega:
        Wave angular frequency in rad/s.
    frequency_index:
        Hydrodynamic frequency index used when the response must be solved.
    wave_direction_deg:
        Wave incidence direction in degrees. The current 10x10 data path only
        contains direction 0 deg; this value is still recorded for traceability.
    """

    omega: float
    frequency_index: int = 0
    wave_direction_deg: float = 0.0
    label: str = "fixed_frequency"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.omega < 0.0:
            raise ValueError("omega must be non-negative rad/s")
        object.__setattr__(self, "omega", float(self.omega))
        object.__setattr__(self, "frequency_index", int(self.frequency_index))
        object.__setattr__(self, "wave_direction_deg", float(self.wave_direction_deg))
        object.__setattr__(self, "meta", dict(self.meta))

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for CSV/JSON summaries."""

        return {
            "omega": self.omega,
            "frequency_index": self.frequency_index,
            "wave_direction_deg": self.wave_direction_deg,
            "scenario_label": self.label,
            **dict(self.meta),
        }


@dataclass(frozen=True)
class PitchStiffnessDesign:
    """Uniform released-pitch stiffness design for the current 10x10 hinge case.

    ``pitch_stiffness`` is the released rotational stiffness used by the hinge
    specs. Translational and non-released rotational components keep
    ``coupling_stiffness``.
    """

    pitch_stiffness: float
    coupling_stiffness: float = 1.0e10
    label: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pitch_stiffness < 0.0:
            raise ValueError("pitch_stiffness must be non-negative")
        if self.coupling_stiffness < 0.0:
            raise ValueError("coupling_stiffness must be non-negative")
        object.__setattr__(self, "pitch_stiffness", float(self.pitch_stiffness))
        object.__setattr__(self, "coupling_stiffness", float(self.coupling_stiffness))
        if not self.label:
            object.__setattr__(self, "label", _stiffness_label(self.pitch_stiffness))
        object.__setattr__(self, "meta", dict(self.meta))

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for CSV/JSON summaries."""

        return {
            "pitch_stiffness": self.pitch_stiffness,
            "pitch_stiffness_label": self.label,
            "coupling_stiffness": self.coupling_stiffness,
            **dict(self.meta),
        }


@dataclass(frozen=True)
class BoundaryStiffnessDesign:
    """Grouped hinge stiffness design for modular-grid internal boundaries."""

    values: tuple[float, ...] | Mapping[str, float]
    grouping: HingeGrouping = "continuous_boundary"
    parameter: HingeParameter = "released_dof_stiffness"
    coupling_stiffness: float = 1.0e10
    label: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.values, Mapping):
            values = {
                str(name): float(value)
                for name, value in self.values.items()
            }
            if any(value < 0.0 for value in values.values()):
                raise ValueError("boundary stiffness values must be non-negative")
            object.__setattr__(self, "values", values)
        else:
            values = tuple(float(value) for value in self.values)
            if any(value < 0.0 for value in values):
                raise ValueError("boundary stiffness values must be non-negative")
            object.__setattr__(self, "values", values)

        if self.coupling_stiffness < 0.0:
            raise ValueError("coupling_stiffness must be non-negative")
        object.__setattr__(self, "coupling_stiffness", float(self.coupling_stiffness))
        if not self.label:
            object.__setattr__(self, "label", f"{self.grouping}_{self.parameter}")
        object.__setattr__(self, "meta", dict(self.meta))

    def values_for_groups(self, group_names: Sequence[str]) -> dict[str, float]:
        """Return design values keyed by hinge design group name."""

        if isinstance(self.values, Mapping):
            missing = [name for name in group_names if name not in self.values]
            if missing:
                raise KeyError(f"Missing boundary stiffness values for groups: {missing}")
            return {name: float(self.values[name]) for name in group_names}
        if len(self.values) != len(group_names):
            raise ValueError(f"Expected {len(group_names)} values, got {len(self.values)}")
        return {
            name: float(value)
            for name, value in zip(group_names, self.values)
        }

    def as_dict(self, group_names: Sequence[str] = ()) -> dict[str, Any]:
        """Return a flat dictionary suitable for CSV/JSON summaries."""

        if group_names:
            values = self.values_for_groups(group_names)
        elif isinstance(self.values, Mapping):
            values = {str(name): float(value) for name, value in self.values.items()}
        else:
            values = {
                f"boundary_{index + 1:02d}": float(value)
                for index, value in enumerate(self.values)
            }
        raw_values = np.asarray(list(values.values()), dtype=float)
        return {
            "design_label": self.label,
            "grouping": self.grouping,
            "stiffness_parameter": self.parameter,
            "design_dimension": len(values),
            "boundary_stiffness_min": float(raw_values.min()) if raw_values.size else 0.0,
            "boundary_stiffness_max": float(raw_values.max()) if raw_values.size else 0.0,
            "boundary_stiffness_mean": float(raw_values.mean()) if raw_values.size else 0.0,
            "coupling_stiffness": self.coupling_stiffness,
            **{
                f"k_{name}": value
                for name, value in values.items()
            },
            **dict(self.meta),
        }


@dataclass(frozen=True)
class SingleFrequencyEvaluation:
    """Response and connector-force metrics for one fixed-frequency design."""

    case_id: str
    design: Mapping[str, Any]
    scenario: Mapping[str, Any]
    metrics: Mapping[str, Any]
    connector_rows: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "design", dict(self.design))
        object.__setattr__(self, "scenario", dict(self.scenario))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(
            self,
            "connector_rows",
            tuple(dict(row) for row in self.connector_rows),
        )

    def summary_row(self) -> dict[str, Any]:
        """Return one flat row combining design, scenario, and metrics."""

        return {
            "case_id": self.case_id,
            **dict(self.design),
            **dict(self.scenario),
            **dict(self.metrics),
        }


def _stiffness_label(value: float) -> str:
    """Return a compact, file-name-safe label for one stiffness value."""

    if value == 0.0:
        return "0"
    return f"{value:g}".replace("+", "")


def _normalize_pitch_design(
    design: PitchStiffnessDesign | Mapping[str, Any],
) -> PitchStiffnessDesign:
    """Accept the public dict form while keeping a typed internal design."""

    if isinstance(design, PitchStiffnessDesign):
        return design
    if "pitch_stiffness" in design:
        pitch_stiffness = design["pitch_stiffness"]
    elif "k_pitch" in design:
        pitch_stiffness = design["k_pitch"]
    else:
        raise KeyError("design must contain 'pitch_stiffness' or 'k_pitch'")
    return PitchStiffnessDesign(
        pitch_stiffness=float(pitch_stiffness),
        coupling_stiffness=float(
            design.get("coupling_stiffness", design.get("k_hinge", 1.0e10))
        ),
        label=str(design.get("pitch_stiffness_label", design.get("label", ""))),
        meta={
            key: value
            for key, value in design.items()
            if key
            not in {
                "pitch_stiffness",
                "k_pitch",
                "coupling_stiffness",
                "k_hinge",
                "pitch_stiffness_label",
                "label",
            }
        },
    )


def _normalize_scenario(
    scenario: SingleFrequencyScenario | Mapping[str, Any],
) -> SingleFrequencyScenario:
    """Accept the public dict form while keeping a typed internal scenario."""

    if isinstance(scenario, SingleFrequencyScenario):
        return scenario
    if "omega" not in scenario:
        raise KeyError("scenario must contain 'omega' in rad/s")
    return SingleFrequencyScenario(
        omega=float(scenario["omega"]),
        frequency_index=int(scenario.get("frequency_index", 0)),
        wave_direction_deg=float(
            scenario.get("wave_direction_deg", scenario.get("direction_deg", 0.0))
        ),
        label=str(scenario.get("scenario_label", scenario.get("label", "fixed_frequency"))),
        meta={
            key: value
            for key, value in scenario.items()
            if key
            not in {
                "omega",
                "frequency_index",
                "wave_direction_deg",
                "direction_deg",
                "scenario_label",
                "label",
            }
        },
    )


def _normalize_boundary_design(
    design: BoundaryStiffnessDesign | Mapping[str, Any],
) -> BoundaryStiffnessDesign:
    """Accept public dict forms for boundary-grouped stiffness designs."""

    if isinstance(design, BoundaryStiffnessDesign):
        return design
    if "boundary_stiffness_values" in design:
        values = design["boundary_stiffness_values"]
    elif "values" in design:
        values = design["values"]
    else:
        prefixed = {
            key.removeprefix("k_"): value
            for key, value in design.items()
            if str(key).startswith("k_")
        }
        if not prefixed:
            raise KeyError(
                "boundary design must contain 'boundary_stiffness_values', "
                "'values', or k_<group_name> entries"
            )
        values = prefixed

    return BoundaryStiffnessDesign(
        values=values,
        grouping=design.get("grouping", "continuous_boundary"),
        parameter=design.get(
            "stiffness_parameter",
            design.get("parameter", "released_dof_stiffness"),
        ),
        coupling_stiffness=float(
            design.get("coupling_stiffness", design.get("k_hinge", 1.0e10))
        ),
        label=str(design.get("design_label", design.get("label", ""))),
        meta={
            key: value
            for key, value in design.items()
            if key
            not in {
                "boundary_stiffness_values",
                "values",
                "grouping",
                "stiffness_parameter",
                "parameter",
                "coupling_stiffness",
                "k_hinge",
                "design_label",
                "label",
            }
            and not str(key).startswith("k_")
        },
    )


def _component_indices(labels: Sequence[str], names: Sequence[str]) -> tuple[int, ...]:
    """Return component indices whose labels are in ``names``."""

    wanted = set(names)
    return tuple(index for index, label in enumerate(labels) if label in wanted)


def _envelope_for_indices(
    values: np.ndarray,
    indices: Sequence[int],
) -> tuple[float, float]:
    """Return harmonic envelope and angle for selected components."""

    if not indices:
        return 0.0, 0.0
    selected = np.asarray(values)[list(indices)]
    envelope, angle = harmonic_vector_norm_envelope(selected)
    return float(envelope), float(angle)


def connector_envelope_rows(
    case,
    response: np.ndarray,
    omega: float,
    *,
    cid_prefix: str = "design",
) -> tuple[dict[str, Any], ...]:
    """Recover connector force envelopes for one case response.

    Parameters
    ----------
    response:
        Complex frequency-domain displacement vector ``x_hat``. Units follow
        the structural model, normally m and rad.
    omega:
        Wave angular frequency in rad/s used in ``K + i omega C``.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One CSV-friendly row per connector node pair. Force entries are
        harmonic envelopes; translational entries are forces and rotational
        entries are moments.
    """

    connectors = build_case_hinge_pair_connectors(case, cid_prefix=cid_prefix)
    recovered = recover_connector_response(
        np.asarray(response).reshape(-1),
        omega=float(omega),
        connectors=connectors,
    )

    rows: list[dict[str, Any]] = []
    for connector in connectors:
        item = recovered[connector.cid]
        force_hat = np.asarray(item["force_hat"]).reshape(-1)
        delta_hat = np.asarray(item["delta_hat"]).reshape(-1)
        labels = connector.labels
        shear_indices = _component_indices(labels, ("uz",))
        bending_indices = _component_indices(labels, ("rx", "ry", "rz"))
        released_indices = tuple(connector.meta.get("released_retained_indices", ()))

        shear_envelope, shear_angle = _envelope_for_indices(force_hat, shear_indices)
        bending_envelope, bending_angle = _envelope_for_indices(force_hat, bending_indices)
        released_envelope, released_angle = _envelope_for_indices(force_hat, released_indices)

        rows.append(
            {
                "cid": connector.cid,
                "hinge_line": connector.meta.get("hinge_line"),
                "hinge_name": connector.meta.get("hinge_name", ""),
                "pair_index": connector.meta.get("pair_index"),
                "node_a": connector.meta.get("node_a"),
                "node_b": connector.meta.get("node_b"),
                "k_hinge": connector.meta.get("k_hinge"),
                "released_dof_stiffness": connector.meta.get("released_dof_stiffness"),
                "released_labels": ";".join(connector.meta.get("released_labels", ())),
                "shear_force_envelope": shear_envelope,
                "shear_force_angle_rad": shear_angle,
                "bending_moment_envelope": bending_envelope,
                "bending_moment_angle_rad": bending_angle,
                "released_moment_envelope": released_envelope,
                "released_moment_angle_rad": released_angle,
                "delta_hat_norm_abs": float(np.linalg.norm(np.abs(delta_hat))),
                "force_hat_max_component_abs": (
                    float(np.abs(force_hat).max()) if force_hat.size else 0.0
                ),
            }
        )

    return tuple(rows)


def heave_amplitude_grid(
    case,
    response: np.ndarray,
    *,
    heave_grid: np.ndarray | None = None,
    merge_interfaces: bool = True,
) -> np.ndarray:
    """Return heave-amplitude grid in m for metrics.

    If a precomputed grid is supplied it is used directly. Otherwise the
    standard 10x10 extractor is used when possible, with a one-row fallback for
    small unit-test cases.
    """

    if heave_grid is not None:
        return np.asarray(heave_grid, dtype=float)

    if hasattr(case, "grid"):
        try:
            return np.asarray(
                extract_complex_hinge_heave_grid(
                    case,
                    response,
                    merge_interfaces=merge_interfaces,
                ),
                dtype=float,
            )
        except Exception:
            # Small tests may define only a minimal grid-like object.
            pass

    response_vector = np.asarray(response).reshape(-1)
    retained_dofs_per_node = int(case.retained_dofs_per_node)
    return np.abs(response_vector[2::retained_dofs_per_node]).reshape(1, -1)


def heave_metrics(
    case,
    response: np.ndarray,
    *,
    heave_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return basic heave-amplitude metrics in m."""

    grid = heave_amplitude_grid(case, response, heave_grid=heave_grid)
    return {
        "heave_grid_shape": tuple(int(value) for value in grid.shape),
        "min_heave": float(np.min(grid)),
        "max_heave": float(np.max(grid)),
        "mean_heave": float(np.mean(grid)),
    }


def _max_row(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    """Return row with maximum ``key`` or a zero placeholder."""

    if not rows:
        return {"cid": "", key: 0.0}
    return max(rows, key=lambda row: float(row[key]))


def evaluate_design_response(
    case,
    response: np.ndarray,
    omega: float,
    *,
    design: Mapping[str, Any] | None = None,
    scenario: Mapping[str, Any] | None = None,
    heave_grid: np.ndarray | None = None,
    cid_prefix: str = "design",
) -> SingleFrequencyEvaluation:
    """Evaluate one solved fixed-frequency design response.

    This is the reusable core for parameter scans: it does not solve the
    hydroelastic system, and therefore it is suitable for cached responses from
    notebooks or scripts.
    """

    connector_rows = connector_envelope_rows(
        case,
        response,
        omega,
        cid_prefix=cid_prefix,
    )
    response_metrics = heave_metrics(case, response, heave_grid=heave_grid)
    max_shear_row = _max_row(connector_rows, "shear_force_envelope")
    max_bending_row = _max_row(connector_rows, "bending_moment_envelope")
    max_released_row = _max_row(connector_rows, "released_moment_envelope")

    metrics = {
        **response_metrics,
        "connector_count": len(connector_rows),
        "max_connector_shear_envelope": float(max_shear_row["shear_force_envelope"]),
        "max_connector_shear_cid": max_shear_row["cid"],
        "max_connector_bending_envelope": float(max_bending_row["bending_moment_envelope"]),
        "max_connector_bending_cid": max_bending_row["cid"],
        "max_released_moment_envelope": float(max_released_row["released_moment_envelope"]),
        "max_released_moment_cid": max_released_row["cid"],
    }
    case_id = getattr(case, "case_id", case.__class__.__name__)
    return SingleFrequencyEvaluation(
        case_id=case_id,
        design={} if design is None else design,
        scenario={"omega": float(omega)} if scenario is None else scenario,
        metrics=metrics,
        connector_rows=connector_rows,
    )


def evaluate_complex_hinge_pitch_design(
    design: PitchStiffnessDesign | Mapping[str, Any],
    scenario: SingleFrequencyScenario | Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
    response: np.ndarray | None = None,
    heave_grid: np.ndarray | None = None,
    solve_if_response_missing: bool = True,
    cid_prefix: str = "pitch_design",
) -> SingleFrequencyEvaluation:
    """Evaluate the current 10x10 case for one uniform pitch stiffness design."""

    design_obj = _normalize_pitch_design(design)
    scenario_obj = _normalize_scenario(scenario)
    case = build_complex_hinge_10x10_case(
        data_root,
        k_hinge=design_obj.coupling_stiffness,
        released_dof_stiffness=design_obj.pitch_stiffness,
    )
    if scenario_obj.frequency_index != case.frequency_index:
        case = replace(case, frequency_index=scenario_obj.frequency_index)

    omega = scenario_obj.omega
    if response is None:
        if not solve_if_response_missing:
            raise ValueError("response is required when solve_if_response_missing is False")
        solved = solve_complex_hinge_case(case)
        response = solved.response
        heave_grid = solved.heave_grid_merged if heave_grid is None else heave_grid
        omega = float(solved.omega)

    return evaluate_design_response(
        case,
        response,
        omega,
        design=design_obj.as_dict(),
        scenario=scenario_obj.as_dict(),
        heave_grid=heave_grid,
        cid_prefix=cid_prefix,
    )


def evaluate_complex_hinge_boundary_design(
    design: BoundaryStiffnessDesign | Mapping[str, Any],
    scenario: SingleFrequencyScenario | Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
    response: np.ndarray | None = None,
    heave_grid: np.ndarray | None = None,
    solve_if_response_missing: bool = True,
    cid_prefix: str = "boundary18_design",
) -> SingleFrequencyEvaluation:
    """Evaluate one grouped-boundary stiffness design for the current 10x10 case."""

    design_obj = _normalize_boundary_design(design)
    scenario_obj = _normalize_scenario(scenario)
    base_case = build_complex_hinge_10x10_case(
        data_root,
        k_hinge=design_obj.coupling_stiffness,
    )
    groups = build_hinge_design_groups(base_case, design_obj.grouping)
    case = apply_grouped_hinge_stiffness(
        base_case,
        groups,
        design_obj.values_for_groups([group.name for group in groups]),
        parameter=design_obj.parameter,
    )
    if scenario_obj.frequency_index != case.frequency_index:
        case = replace(case, frequency_index=scenario_obj.frequency_index)

    omega = scenario_obj.omega
    if response is None:
        if not solve_if_response_missing:
            raise ValueError("response is required when solve_if_response_missing is False")
        solved = solve_complex_hinge_case(case)
        response = solved.response
        heave_grid = solved.heave_grid_merged if heave_grid is None else heave_grid
        omega = float(solved.omega)

    return evaluate_design_response(
        case,
        response,
        omega,
        design=design_obj.as_dict([group.name for group in groups]),
        scenario=scenario_obj.as_dict(),
        heave_grid=heave_grid,
        cid_prefix=cid_prefix,
    )


def evaluate_design(
    design: PitchStiffnessDesign | BoundaryStiffnessDesign | Mapping[str, Any],
    scenario: SingleFrequencyScenario | Mapping[str, Any],
    *,
    case_type: str = "complex_hinge_10x10_pitch",
    data_root: str | Path | None = None,
    response: np.ndarray | None = None,
    heave_grid: np.ndarray | None = None,
    solve_if_response_missing: bool = True,
    cid_prefix: str = "design",
) -> SingleFrequencyEvaluation:
    """Public fixed-frequency evaluator.

    The current implementation supports the 10x10 modular hinge pitch-stiffness
    study. The function shape is intentionally generic so later wind/PV cases
    can plug in without changing notebooks or parameter-scan scripts.
    """

    supported = {
        "complex_hinge_10x10",
        "complex_hinge_10x10_pitch",
        "complex_hinge_10x10_boundary18",
    }
    if case_type not in supported:
        raise ValueError(
            f"Unsupported case_type {case_type!r}; expected one of {sorted(supported)}"
        )
    if case_type == "complex_hinge_10x10_boundary18":
        return evaluate_complex_hinge_boundary_design(
            design,
            scenario,
            data_root=data_root,
            response=response,
            heave_grid=heave_grid,
            solve_if_response_missing=solve_if_response_missing,
            cid_prefix=cid_prefix,
        )
    return evaluate_complex_hinge_pitch_design(
        design,
        scenario,
        data_root=data_root,
        response=response,
        heave_grid=heave_grid,
        solve_if_response_missing=solve_if_response_missing,
        cid_prefix=cid_prefix,
    )
