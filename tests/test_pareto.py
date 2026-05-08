from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    MetricConstraint,
    MetricObjective,
    constraint_margins,
    constraints_satisfied,
    mark_pareto_rows,
    pareto_mask_from_values,
)


class ParetoTests(unittest.TestCase):
    def test_pareto_mask_for_two_minimization_objectives(self) -> None:
        values = np.array(
            [
                [1.0, 5.0],
                [2.0, 6.0],
                [3.0, 2.0],
                [1.5, 3.0],
            ]
        )

        mask = pareto_mask_from_values(values)

        np.testing.assert_array_equal(mask, [True, False, True, True])

    def test_constraint_margin_and_satisfaction(self) -> None:
        row = {"max_connector_bending_envelope": 8.0, "mean_heave": 0.9}
        constraints = (
            MetricConstraint(
                "bending_limit",
                "max_connector_bending_envelope",
                upper_bound=10.0,
            ),
            MetricConstraint("heave_floor", "mean_heave", lower_bound=0.8),
        )

        margins = constraint_margins(row, constraints)

        self.assertTrue(constraints_satisfied(row, constraints))
        np.testing.assert_allclose(margins["bending_limit"], 2.0)
        np.testing.assert_allclose(margins["heave_floor"], 0.1)

    def test_mark_pareto_rows_respects_feasibility(self) -> None:
        rows = [
            {"design": "a", "mean_heave": 1.0, "max_connector_bending_envelope": 1.0},
            {"design": "b", "mean_heave": 0.8, "max_connector_bending_envelope": 2.0},
            {"design": "c", "mean_heave": 0.7, "max_connector_bending_envelope": 5.0},
        ]
        objectives = (
            MetricObjective("mean_heave", "mean_heave"),
            MetricObjective("bending", "max_connector_bending_envelope"),
        )
        constraints = (
            MetricConstraint(
                "bending_limit",
                "max_connector_bending_envelope",
                upper_bound=3.0,
            ),
        )

        marked = mark_pareto_rows(rows, objectives, constraints)

        self.assertTrue(marked[0]["is_feasible"])
        self.assertTrue(marked[1]["is_feasible"])
        self.assertFalse(marked[2]["is_feasible"])
        self.assertTrue(marked[0]["is_pareto"])
        self.assertTrue(marked[1]["is_pareto"])
        self.assertFalse(marked[2]["is_pareto"])


if __name__ == "__main__":
    unittest.main()
