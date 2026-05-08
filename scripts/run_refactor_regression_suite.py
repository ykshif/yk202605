"""Run the RODM refactor regression suite from the repository root."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ("baseline verification", [sys.executable, "scripts/verify_reference_case_300.py"]),
    ("RODM case orchestration", [sys.executable, "scripts/validate_rodm_case_orchestration.py"]),
    ("reduction and solver kernels", [sys.executable, "scripts/validate_reduction_solver_kernels.py"]),
    ("structure connectors", [sys.executable, "scripts/validate_structure_connectors.py"]),
    ("environment/load/power/strength helpers", [sys.executable, "scripts/validate_environment_load_power_strength.py"]),
    ("RODM staged pipeline", [sys.executable, "scripts/validate_rodm_pipeline_stages.py"]),
    ("config-driven reference case", [sys.executable, "scripts/validate_config_driven_reference_case.py"]),
    ("reference case workflow", [sys.executable, "scripts/run_reference_case_300_workflow.py"]),
    ("legacy vs packaged RODM", [sys.executable, "scripts/compare_legacy_and_packaged_rodm.py"]),
    ("hinge model validation", [sys.executable, "scripts/validate_hinge_model.py"]),
]


def main() -> int:
    started = time.perf_counter()
    for label, command in COMMANDS:
        print(f"\n=== {label} ===")
        result = subprocess.run(command, cwd=REPO_ROOT, text=True)
        if result.returncode != 0:
            print(f"FAILED: {label}")
            return result.returncode

    elapsed = time.perf_counter() - started
    print(f"\nRefactor regression suite passed in {elapsed:.3f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
