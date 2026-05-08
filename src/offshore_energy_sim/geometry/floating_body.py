"""Geometry models for floating-body reference cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectangularFloatingBody:
    """Rectangular floating body dimensions and mesh metadata."""

    length_m: float
    width_m: float
    thickness_m: float | None = None
    total_nodes: int | None = None
    retained_dofs_per_node: int | None = None
    mesh_label: str | None = None

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def aspect_ratio(self) -> float:
        return self.length_m / self.width_m
