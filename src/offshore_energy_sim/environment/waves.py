"""Wave-environment value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegularWave:
    """Regular wave descriptor used by frequency-domain validation cases."""

    wavelength_m: float
    direction_deg: float
    amplitude_m: float | None = None

    @property
    def heading_rad(self) -> float:
        return self.direction_deg * 3.141592653589793 / 180.0
