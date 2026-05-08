"""Validate read-only input files for the 300 m x 60 m reference case."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    default_paths,
    format_input_summary_report,
    verify_hashes,
)


def main() -> int:
    paths = default_paths(REPO_ROOT)
    print(format_input_summary_report(paths))
    return 0 if not verify_hashes(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
