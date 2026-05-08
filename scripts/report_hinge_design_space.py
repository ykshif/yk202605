"""Report hinge design-space dimensions for the 10x10 modular case."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    build_hinge_design_groups,
    summarize_hinge_design_space,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
)


OUTPUT_ROOT = REPO_ROOT / "results" / "hinge_design_space"


def main() -> None:
    case = build_complex_hinge_10x10_case("/Users/yongkang/data/DM-FEM2D")
    summary = summarize_hinge_design_space(case)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_ROOT / "hinge_design_space_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary.as_dict(), file, indent=2)

    groups_path = OUTPUT_ROOT / "continuous_boundary_groups.csv"
    groups = build_hinge_design_groups(case, "continuous_boundary")
    with groups_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["name", "orientation", "hinge_line_count", "hinge_indices"],
        )
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "name": group.name,
                    "orientation": group.orientation,
                    "hinge_line_count": len(group.hinge_indices),
                    "hinge_indices": " ".join(str(index + 1) for index in group.hinge_indices),
                }
            )

    print(f"summary_path: {summary_path}")
    print(f"groups_path: {groups_path}")
    print(f"hinge_line_count: {summary.hinge_line_count}")
    print(f"connector_pair_count: {summary.connector_pair_count}")
    print(f"continuous_boundary_dimension: {summary.continuous_boundary_dimension}")


if __name__ == "__main__":
    main()
