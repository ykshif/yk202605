"""Compare legacy DM_Method RODM output with the packaged RODM solver."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import DM_Method  # noqa: E402
from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    build_rodm_frequency_case,
    default_paths,
)
from offshore_energy_sim.solver import solve_rodm_frequency_case  # noqa: E402


RESULT_DIR = REPO_ROOT / "results"
LEGACY_RESPONSE_PATH = RESULT_DIR / "reference_case_300_legacy_dm_method.npy"


def compare(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    diff = actual - expected
    expected_norm = np.linalg.norm(expected)
    return {
        "name": name,
        "actual_shape": tuple(actual.shape),
        "expected_shape": tuple(expected.shape),
        "max_abs_error": float(np.max(np.abs(diff))),
        "l2_relative_error": float(np.linalg.norm(diff) / expected_norm) if expected_norm else float("nan"),
    }


def print_metrics(metrics: dict[str, object]) -> None:
    print(metrics["name"])
    for key, value in metrics.items():
        if key != "name":
            print(f"  {key}: {value}")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    paths = default_paths(REPO_ROOT)
    structure_paths = {
        "mass": str(paths.structural_mass_file),
        "stiffness": str(paths.structural_stiffness_file),
    }

    start = time.perf_counter()
    legacy = DM_Method.perform_RODM_reduce_order_model(
        num_nodes=793,
        node_position_params=(424, 6, 10),
        hydrodynamic_data_path=str(paths.hydrodynamic_file),
        structure_data_paths=structure_paths,
        use_hydrostatic=True,
    )
    legacy_elapsed = time.perf_counter() - start
    np.save(LEGACY_RESPONSE_PATH, legacy)

    start = time.perf_counter()
    packaged = solve_rodm_frequency_case(build_rodm_frequency_case(paths)).global_displacement
    packaged_elapsed = time.perf_counter() - start

    baseline = np.load(paths.response_file)

    print(f"legacy_elapsed_seconds: {legacy_elapsed:.3f}")
    print(f"packaged_elapsed_seconds: {packaged_elapsed:.3f}")
    print()
    print_metrics(compare("packaged_vs_legacy", packaged, legacy))
    print()
    print_metrics(compare("legacy_vs_baseline", legacy, baseline))
    print()
    print_metrics(compare("packaged_vs_baseline", packaged, baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
