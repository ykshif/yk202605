"""Verify the 300 m x 60 m floating-body reference case.

This legacy-compatible entry point is intentionally read-only. The reusable
implementation lives in ``offshore_energy_sim.postprocess.reference_case_300``.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    default_paths,
    format_verification_report,
    verify_reference_case_300,
)


def main() -> int:
    result = verify_reference_case_300(default_paths(REPO_ROOT))
    print(format_verification_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
