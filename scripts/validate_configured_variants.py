"""Validate explicitly configured RODM case variants."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import build_rodm_frequency_case_from_config  # noqa: E402
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


def _assert_response_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    difference = actual - expected
    max_abs_error = float(np.max(np.abs(difference)))
    l2_relative_error = float(np.linalg.norm(difference) / np.linalg.norm(expected))
    print(label)
    print(f"  actual_shape: {actual.shape}")
    print(f"  expected_shape: {expected.shape}")
    print(f"  max_abs_error: {max_abs_error}")
    print(f"  l2_relative_error: {l2_relative_error}")
    if max_abs_error != 0.0:
        raise AssertionError(f"{label} changed numerical results")


def main() -> int:
    reversed_case = build_rodm_frequency_case_from_config(
        REPO_ROOT / "configs" / "reference_case_300_hydro_reversed.yaml",
    )
    if not reversed_case.reverse_hydrodynamic_node_order:
        raise AssertionError("Hydro-reversed config did not enable node reversal.")

    result = solve_rodm_frequency_case(reversed_case)
    expected = np.load(REPO_ROOT / "results" / "reference_case_300_rodm_hydro_reversed.npy")
    _assert_response_equal(
        "hydro_reversed_config_vs_candidate",
        result.global_displacement,
        expected,
    )
    print("Configured variant validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
