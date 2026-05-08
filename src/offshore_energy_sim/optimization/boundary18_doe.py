"""DOE sample generation for 18-variable boundary hinge stiffness studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOUNDARY18_GROUP_NAMES = tuple(
    [f"x_boundary_{index:02d}" for index in range(1, 10)]
    + [f"y_boundary_{index:02d}" for index in range(1, 10)]
)


@dataclass(frozen=True)
class Boundary18Sample:
    """One candidate 18-variable boundary-stiffness design."""

    name: str
    values: tuple[float, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.values) != 18:
            raise ValueError(f"Boundary18Sample expects 18 values, got {len(self.values)}")
        if any(value < 0.0 for value in self.values):
            raise ValueError("Boundary stiffness values must be non-negative")

    def value_by_group(self) -> dict[str, float]:
        """Return values keyed by standard boundary group name."""

        return {
            name: float(value)
            for name, value in zip(BOUNDARY18_GROUP_NAMES, self.values)
        }


def _geometric_mid(low: float, high: float) -> float:
    """Return geometric mean for positive stiffness bounds."""

    if low <= 0.0 or high <= 0.0:
        raise ValueError("DOE stiffness bounds must be positive")
    return float(np.sqrt(low * high))


def _center_profile(low: float, high: float) -> tuple[float, ...]:
    """Return nine values, stiffest at the middle boundary."""

    center = 5
    max_distance = 4
    values = []
    for boundary_index in range(1, 10):
        distance = abs(boundary_index - center)
        weight = 1.0 - distance / max_distance
        values.append(low * (high / low) ** weight)
    return tuple(float(value) for value in values)


def _edge_profile(low: float, high: float) -> tuple[float, ...]:
    """Return nine values, stiffest at the outer internal boundaries."""

    center = 5
    max_distance = 4
    values = []
    for boundary_index in range(1, 10):
        distance = abs(boundary_index - center)
        weight = distance / max_distance
        values.append(low * (high / low) ** weight)
    return tuple(float(value) for value in values)


def _gradient_profile(low: float, high: float) -> tuple[float, ...]:
    """Return nine log-spaced values from low to high."""

    return tuple(float(value) for value in np.geomspace(low, high, 9))


def _safe_label(value: float) -> str:
    """Return a compact file-name-safe numeric label."""

    return f"{float(value):.2e}".replace("+", "").replace(".", "p")


def generate_boundary18_doe_samples(
    *,
    low: float = 1.0e8,
    high: float = 1.0e9,
    random_count: int = 4,
    seed: int = 20260502,
) -> tuple[Boundary18Sample, ...]:
    """Return deterministic structured plus random 18-variable DOE samples."""

    if low <= 0.0 or high <= 0.0:
        raise ValueError("DOE stiffness bounds must be positive")
    if low > high:
        raise ValueError("low cannot exceed high")
    if random_count < 0:
        raise ValueError("random_count must be non-negative")

    mid = _geometric_mid(low, high)
    center = _center_profile(low, high)
    edge = _edge_profile(low, high)
    gradient = _gradient_profile(low, high)
    reverse_gradient = tuple(reversed(gradient))

    samples = [
        Boundary18Sample("uniform_low", tuple([low] * 18), "All boundaries use low stiffness."),
        Boundary18Sample("uniform_mid", tuple([mid] * 18), "All boundaries use geometric-mid stiffness."),
        Boundary18Sample("uniform_high", tuple([high] * 18), "All boundaries use high stiffness."),
        Boundary18Sample("center_stiff", center + center, "Center x/y boundaries are stiffest."),
        Boundary18Sample("edge_stiff", edge + edge, "Outer internal x/y boundaries are stiffest."),
        Boundary18Sample("x_high_y_low", tuple([high] * 9 + [low] * 9), "x boundaries high, y boundaries low."),
        Boundary18Sample("x_low_y_high", tuple([low] * 9 + [high] * 9), "x boundaries low, y boundaries high."),
        Boundary18Sample("x_gradient_y_mid", gradient + tuple([mid] * 9), "x boundaries increase, y boundaries mid."),
        Boundary18Sample("x_mid_y_gradient", tuple([mid] * 9) + gradient, "x boundaries mid, y boundaries increase."),
        Boundary18Sample(
            "opposed_gradients",
            gradient + reverse_gradient,
            "x boundaries increase while y boundaries decrease.",
        ),
    ]

    rng = np.random.default_rng(seed)
    log_low = np.log10(low)
    log_high = np.log10(high)
    for index in range(random_count):
        values = 10.0 ** rng.uniform(log_low, log_high, size=18)
        samples.append(
            Boundary18Sample(
                f"random_log_{index + 1:02d}",
                tuple(float(value) for value in values),
                "Random log-uniform boundary stiffness sample.",
            )
        )

    return tuple(samples)


def generate_boundary18_refined_samples(
    *,
    low: float = 1.0e8,
    high: float = 1.0e9,
) -> tuple[Boundary18Sample, ...]:
    """Return local refinement samples around promising boundary18 regions.

    The refined set focuses on three regions seen in the first DOE:

    * orientation-asymmetric designs near ``x_high_y_low``;
    * center-stiff designs with different low/high contrast;
    * uniform designs around the geometric middle.
    """

    if low <= 0.0 or high <= 0.0:
        raise ValueError("DOE stiffness bounds must be positive")
    if low > high:
        raise ValueError("low cannot exceed high")

    mid = _geometric_mid(low, high)
    samples: list[Boundary18Sample] = [
        Boundary18Sample("anchor_uniform_mid", tuple([mid] * 18), "Anchor near balanced uniform design."),
        Boundary18Sample("anchor_x_high_y_low", tuple([high] * 9 + [low] * 9), "Anchor x-high/y-low design."),
        Boundary18Sample("anchor_center_stiff", _center_profile(low, high) * 2, "Anchor center-stiff design."),
    ]

    for value in (2.0e8, 5.0e8, 7.0e8):
        clipped = min(max(value, low), high)
        samples.append(
            Boundary18Sample(
                f"uniform_{_safe_label(clipped)}",
                tuple([clipped] * 18),
                "Uniform stiffness refinement around the middle range.",
            )
        )

    x_levels = (5.0e8, 7.5e8, high)
    y_levels = (low, 1.8e8, 3.0e8)
    for x_value in x_levels:
        for y_value in y_levels:
            x_clipped = min(max(x_value, low), high)
            y_clipped = min(max(y_value, low), high)
            if x_clipped == high and y_clipped == low:
                continue
            samples.append(
                Boundary18Sample(
                    f"orient_x_{_safe_label(x_clipped)}_y_{_safe_label(y_clipped)}",
                    tuple([x_clipped] * 9 + [y_clipped] * 9),
                    "Orientation-asymmetric refinement near x-high/y-low.",
                )
            )

    center_bounds = (
        (low, 6.0e8),
        (low, 8.0e8),
        (1.8e8, 8.0e8),
        (1.8e8, high),
    )
    for profile_low, profile_high in center_bounds:
        clipped_low = min(max(profile_low, low), high)
        clipped_high = min(max(profile_high, low), high)
        profile = _center_profile(clipped_low, clipped_high)
        samples.append(
            Boundary18Sample(
                f"center_{_safe_label(clipped_low)}_{_safe_label(clipped_high)}",
                profile + profile,
                "Center-stiff refinement with adjusted contrast.",
            )
        )

    return tuple(samples)
