"""Validate YAML-config-driven construction of the 300 m RODM reference case."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import build_rodm_frequency_case_from_config  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    build_rodm_frequency_case,
    default_paths,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


def _assert_case_equal(label: str, actual, expected) -> None:
    actual_dict = asdict(actual)
    expected_dict = asdict(expected)
    print(label)
    print(f"  actual: {actual_dict}")
    print(f"  expected: {expected_dict}")
    if actual_dict != expected_dict:
        raise AssertionError(f"{label} differs from hard-coded builder")


def _assert_response_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    diff = actual - expected
    max_abs_error = float(np.max(np.abs(diff)))
    l2_relative_error = float(np.linalg.norm(diff) / np.linalg.norm(expected))
    print(label)
    print(f"  actual_shape: {actual.shape}")
    print(f"  expected_shape: {expected.shape}")
    print(f"  max_abs_error: {max_abs_error}")
    print(f"  l2_relative_error: {l2_relative_error}")
    if max_abs_error != 0.0:
        raise AssertionError(f"{label} changed numerical results")


def main() -> int:
    paths = default_paths(REPO_ROOT)
    config_path = REPO_ROOT / "configs" / "reference_case_300.yaml"
    config_case = build_rodm_frequency_case_from_config(config_path)
    hard_coded_case = build_rodm_frequency_case(paths)

    _assert_case_equal("config_case_vs_hard_coded_case_default", config_case, hard_coded_case)

    response = solve_rodm_frequency_case(config_case).global_displacement
    expected_response = np.load(REPO_ROOT / "results" / "reference_case_300_rodm_generated.npy")
    _assert_response_equal("config_driven_response_vs_pre_refactor_output_default", response, expected_response)

    print("Config-driven reference case validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
