"""Plot the 300 m x 60 m reference-case heave RAO comparison.

This legacy-compatible entry point is read-only with respect to numerical data.
The reusable implementation lives in ``offshore_energy_sim.postprocess``.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.postprocess.reference_case_300 import (  # noqa: E402
    default_paths,
    plot_reference_case_300,
)


def main() -> int:
    png_path, pdf_path = plot_reference_case_300(default_paths(REPO_ROOT))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
