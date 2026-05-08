from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    SingleFrequencyScenario,
    evaluate_design,
    evaluate_design_response,
)
from offshore_energy_sim.structure import ExplicitHingeSpec  # noqa: E402


class _TinyGrid:
    total_nodes = 2


class _TinyCase:
    case_id = "tiny_hinge_case"
    grid = _TinyGrid()
    retained_dofs_per_node = 5
    removed_full_dofs_zero_based = (5,)
    hinges = (
        ExplicitHingeSpec(
            nodes_side_a_one_based=(1,),
            nodes_side_b_one_based=(2,),
            k_hinge=1000.0,
            dofs_per_node=6,
            released_dofs_zero_based=(4,),
            released_dof_stiffness=20.0,
            name="tiny pitch hinge",
        ),
    )


class DesignEvaluatorTests(unittest.TestCase):
    def test_evaluate_design_response_reports_heave_and_connector_metrics(self) -> None:
        response = np.array(
            [
                0.0,
                0.0,
                0.01,
                0.20,
                0.50,
                0.0,
                0.0,
                0.00,
                0.10,
                0.00,
            ],
            dtype=np.complex128,
        )

        evaluation = evaluate_design_response(
            _TinyCase(),
            response,
            omega=0.5851,
            design={"pitch_stiffness": 20.0},
            scenario=SingleFrequencyScenario(omega=0.5851).as_dict(),
            heave_grid=np.array([[0.01, 0.00]]),
            cid_prefix="unit",
        )

        row = evaluation.summary_row()
        self.assertEqual(row["case_id"], "tiny_hinge_case")
        self.assertEqual(row["connector_count"], 1)
        np.testing.assert_allclose(row["max_heave"], 0.01)
        np.testing.assert_allclose(row["mean_heave"], 0.005)
        np.testing.assert_allclose(row["max_connector_shear_envelope"], 10.0)
        np.testing.assert_allclose(
            row["max_connector_bending_envelope"],
            np.sqrt(100.0**2 + 10.0**2),
        )
        np.testing.assert_allclose(row["max_released_moment_envelope"], 10.0)

        connector_row = evaluation.connector_rows[0]
        self.assertEqual(connector_row["cid"], "unit_001_001")
        self.assertEqual(connector_row["released_labels"], "ry")

    def test_zero_released_pitch_stiffness_keeps_released_moment_zero(self) -> None:
        class IdealTinyCase(_TinyCase):
            hinges = (
                ExplicitHingeSpec(
                    nodes_side_a_one_based=(1,),
                    nodes_side_b_one_based=(2,),
                    k_hinge=1000.0,
                    dofs_per_node=6,
                    released_dofs_zero_based=(4,),
                    released_dof_stiffness=0.0,
                    name="ideal pitch hinge",
                ),
            )

        response = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        evaluation = evaluate_design_response(
            IdealTinyCase(),
            response,
            omega=0.5851,
            heave_grid=np.array([[0.0, 0.0]]),
        )

        np.testing.assert_allclose(
            evaluation.metrics["max_released_moment_envelope"],
            0.0,
        )

    def test_boundary18_design_evaluator_accepts_cached_response(self) -> None:
        from offshore_energy_sim.validation.complex_hinge_10x10 import (
            build_complex_hinge_10x10_case,
        )

        case = build_complex_hinge_10x10_case("/Users/yongkang/data/DM-FEM2D")
        response = np.zeros(case.grid.total_nodes * case.retained_dofs_per_node)
        heave_grid = np.zeros((61, 61))

        evaluation = evaluate_design(
            {
                "boundary_stiffness_values": tuple(1.0e8 for _ in range(18)),
                "design_label": "unit_boundary18",
            },
            {"omega": 0.5851},
            case_type="complex_hinge_10x10_boundary18",
            response=response,
            heave_grid=heave_grid,
            solve_if_response_missing=False,
        )

        row = evaluation.summary_row()
        self.assertEqual(row["design_dimension"], 18)
        self.assertEqual(row["connector_count"], 1260)
        np.testing.assert_allclose(row["max_heave"], 0.0)
        np.testing.assert_allclose(row["max_connector_bending_envelope"], 0.0)


if __name__ == "__main__":
    unittest.main()
