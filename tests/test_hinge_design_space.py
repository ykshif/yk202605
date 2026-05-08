from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    summarize_hinge_design_space,
)
from offshore_energy_sim.validation.complex_hinge_10x10 import (  # noqa: E402
    build_complex_hinge_10x10_case,
)


class HingeDesignSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_complex_hinge_10x10_case("/Users/yongkang/data/DM-FEM2D")

    def test_10x10_hinge_design_space_counts(self) -> None:
        summary = summarize_hinge_design_space(self.case)

        self.assertEqual(summary.hinge_line_count, 180)
        self.assertEqual(summary.x_hinge_line_count, 90)
        self.assertEqual(summary.y_hinge_line_count, 90)
        self.assertEqual(summary.connector_pair_count, 1260)
        self.assertEqual(summary.pairs_per_hinge_line, 7)
        self.assertEqual(summary.uniform_dimension, 1)
        self.assertEqual(summary.orientation_dimension, 2)
        self.assertEqual(summary.continuous_boundary_dimension, 18)
        self.assertEqual(summary.segment_line_dimension, 180)
        self.assertEqual(summary.connector_pair_dimension, 1260)

    def test_continuous_boundary_grouping_has_18_groups(self) -> None:
        groups = build_hinge_design_groups(self.case, "continuous_boundary")

        self.assertEqual(len(groups), 18)
        self.assertEqual(groups[0].name, "x_boundary_01")
        self.assertEqual(len(groups[0].hinge_indices), 10)
        self.assertEqual(groups[9].name, "y_boundary_01")
        self.assertEqual(len(groups[9].hinge_indices), 10)

    def test_apply_grouped_released_stiffness(self) -> None:
        groups = build_hinge_design_groups(self.case, "orientation")
        modified = apply_grouped_hinge_stiffness(
            self.case,
            groups,
            {"x_hinges": 1.0e8, "y_hinges": 2.0e8},
        )

        x_values = {
            hinge.released_dof_stiffness
            for hinge in modified.hinges
            if hinge.name.startswith("x ")
        }
        y_values = {
            hinge.released_dof_stiffness
            for hinge in modified.hinges
            if hinge.name.startswith("y ")
        }
        self.assertEqual(x_values, {1.0e8})
        self.assertEqual(y_values, {2.0e8})
        self.assertEqual(
            {
                hinge.released_dof_stiffness
                for hinge in self.case.hinges
            },
            {10.0},
        )


if __name__ == "__main__":
    unittest.main()
