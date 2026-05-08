"""Design-space helpers for modular-grid hinge stiffness studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal


HingeGrouping = Literal[
    "uniform",
    "orientation",
    "continuous_boundary",
    "segment_line",
    "connector_pair",
]
HingeParameter = Literal["released_dof_stiffness", "k_hinge"]


@dataclass(frozen=True)
class HingeDesignGroup:
    """A group of hinge lines sharing one design variable."""

    name: str
    hinge_indices: tuple[int, ...]
    orientation: str
    description: str = ""


@dataclass(frozen=True)
class HingeDesignSpaceSummary:
    """Counts for one modular-grid hinge design space."""

    hinge_line_count: int
    x_hinge_line_count: int
    y_hinge_line_count: int
    connector_pair_count: int
    pairs_per_hinge_line: int
    uniform_dimension: int
    orientation_dimension: int
    continuous_boundary_dimension: int
    segment_line_dimension: int
    connector_pair_dimension: int

    def as_dict(self) -> dict[str, int]:
        """Return a CSV/JSON-friendly dictionary."""

        return {
            "hinge_line_count": self.hinge_line_count,
            "x_hinge_line_count": self.x_hinge_line_count,
            "y_hinge_line_count": self.y_hinge_line_count,
            "connector_pair_count": self.connector_pair_count,
            "pairs_per_hinge_line": self.pairs_per_hinge_line,
            "uniform_dimension": self.uniform_dimension,
            "orientation_dimension": self.orientation_dimension,
            "continuous_boundary_dimension": self.continuous_boundary_dimension,
            "segment_line_dimension": self.segment_line_dimension,
            "connector_pair_dimension": self.connector_pair_dimension,
        }


def _hinge_orientation(hinge) -> str:
    """Infer hinge orientation from the standard generated hinge name."""

    name = getattr(hinge, "name", "").lower()
    if name.startswith("x "):
        return "x"
    if name.startswith("y "):
        return "y"
    return "unknown"


def summarize_hinge_design_space(case) -> HingeDesignSpaceSummary:
    """Return hinge-line and connector-pair counts for a standard case."""

    hinges = tuple(case.hinges)
    line_count = len(hinges)
    pair_counts = [len(hinge.node_pairs_one_based) for hinge in hinges]
    pairs_per_line = pair_counts[0] if pair_counts else 0
    if pair_counts and any(count != pairs_per_line for count in pair_counts):
        pairs_per_line = -1

    x_count = sum(1 for hinge in hinges if _hinge_orientation(hinge) == "x")
    y_count = sum(1 for hinge in hinges if _hinge_orientation(hinge) == "y")
    modules_per_side = int(case.grid.modules_per_side)

    return HingeDesignSpaceSummary(
        hinge_line_count=line_count,
        x_hinge_line_count=x_count,
        y_hinge_line_count=y_count,
        connector_pair_count=sum(pair_counts),
        pairs_per_hinge_line=pairs_per_line,
        uniform_dimension=1 if line_count else 0,
        orientation_dimension=sum(1 for count in (x_count, y_count) if count > 0),
        continuous_boundary_dimension=2 * (modules_per_side - 1),
        segment_line_dimension=line_count,
        connector_pair_dimension=sum(pair_counts),
    )


def build_hinge_design_groups(
    case,
    grouping: HingeGrouping,
) -> tuple[HingeDesignGroup, ...]:
    """Build hinge-line design groups for a modular-grid case.

    ``connector_pair`` is counted by :func:`summarize_hinge_design_space`, but
    it is intentionally not returned here because pair-level stiffness requires
    splitting hinge specs into individual node-pair connectors.
    """

    hinges = tuple(case.hinges)
    if grouping == "connector_pair":
        raise ValueError(
            "connector_pair grouping is counted but not applied to hinge-line specs; "
            "use segment_line or lower-dimensional groups first"
        )
    if grouping == "uniform":
        return (
            HingeDesignGroup(
                name="all_hinges",
                hinge_indices=tuple(range(len(hinges))),
                orientation="all",
                description="All x/y hinge lines share one stiffness.",
            ),
        )
    if grouping == "orientation":
        groups: list[HingeDesignGroup] = []
        for orientation in ("x", "y"):
            indices = tuple(
                index
                for index, hinge in enumerate(hinges)
                if _hinge_orientation(hinge) == orientation
            )
            if indices:
                groups.append(
                    HingeDesignGroup(
                        name=f"{orientation}_hinges",
                        hinge_indices=indices,
                        orientation=orientation,
                        description=f"All {orientation}-oriented hinge lines.",
                    )
                )
        return tuple(groups)
    if grouping == "segment_line":
        return tuple(
            HingeDesignGroup(
                name=f"hinge_line_{index + 1:03d}",
                hinge_indices=(index,),
                orientation=_hinge_orientation(hinge),
                description=getattr(hinge, "name", ""),
            )
            for index, hinge in enumerate(hinges)
        )
    if grouping == "continuous_boundary":
        return _continuous_boundary_groups(case)
    raise ValueError(f"Unsupported grouping: {grouping!r}")


def _continuous_boundary_groups(case) -> tuple[HingeDesignGroup, ...]:
    """Group 30 m segment hinges into full-length internal grid boundaries."""

    n = int(case.grid.modules_per_side)
    x_count = n * (n - 1)
    expected_line_count = 2 * x_count
    if len(case.hinges) != expected_line_count:
        raise ValueError(
            "continuous_boundary grouping expects a square generated x/y hinge list"
        )

    groups: list[HingeDesignGroup] = []
    for boundary_column in range(1, n):
        indices = tuple((row - 1) * (n - 1) + (boundary_column - 1) for row in range(1, n + 1))
        groups.append(
            HingeDesignGroup(
                name=f"x_boundary_{boundary_column:02d}",
                hinge_indices=indices,
                orientation="x",
                description="Vertical internal boundary assembled from x hinge segments.",
            )
        )

    y_offset = x_count
    for boundary_row in range(1, n):
        indices = tuple(
            y_offset + (boundary_row - 1) * n + (column - 1)
            for column in range(1, n + 1)
        )
        groups.append(
            HingeDesignGroup(
                name=f"y_boundary_{boundary_row:02d}",
                hinge_indices=indices,
                orientation="y",
                description="Horizontal internal boundary assembled from y hinge segments.",
            )
        )
    return tuple(groups)


def apply_grouped_hinge_stiffness(
    case,
    groups: Sequence[HingeDesignGroup],
    values: Mapping[str, float] | Sequence[float],
    *,
    parameter: HingeParameter = "released_dof_stiffness",
):
    """Return a case copy with grouped hinge stiffness values applied."""

    if isinstance(values, Mapping):
        value_by_name = {str(name): float(value) for name, value in values.items()}
    else:
        value_list = list(values)
        if len(value_list) != len(groups):
            raise ValueError(f"Expected {len(groups)} values, got {len(value_list)}")
        value_by_name = {
            group.name: float(value)
            for group, value in zip(groups, value_list)
        }

    hinges = list(case.hinges)
    assigned: set[int] = set()
    for group in groups:
        if group.name not in value_by_name:
            raise KeyError(f"Missing stiffness value for group {group.name!r}")
        value = value_by_name[group.name]
        if value < 0.0:
            raise ValueError("hinge stiffness values must be non-negative")
        for index in group.hinge_indices:
            if index < 0 or index >= len(hinges):
                raise IndexError(f"Hinge index {index} is outside the case hinge list")
            if index in assigned:
                raise ValueError(f"Hinge index {index} is assigned by more than one group")
            hinges[index] = replace(hinges[index], **{parameter: value})
            assigned.add(index)

    return replace(case, hinges=tuple(hinges))
