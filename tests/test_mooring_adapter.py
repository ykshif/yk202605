from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    build_corner_mooring_reduced_stiffness,
    corner_mooring_diagonal_stiffness,
    corner_node_ids_for_regular_grid,
    project_diagonal_stiffness_to_reduced,
)


class MooringAdapterTests(unittest.TestCase):
    def test_corner_node_ids_for_300m_grid(self) -> None:
        self.assertEqual(corner_node_ids_for_regular_grid(61, 13), (1, 61, 733, 793))

    def test_corner_mooring_diagonal_sets_requested_dofs(self) -> None:
        diagonal = corner_mooring_diagonal_stiffness(
            total_nodes=6,
            retained_dofs_per_node=5,
            nodes_per_x=3,
            nodes_per_y=2,
            surge_stiffness=10.0,
            sway_stiffness=20.0,
            heave_stiffness=0.0,
        )
        expected = np.zeros(30)
        for node in (1, 3, 4, 6):
            expected[(node - 1) * 5 + 0] = 10.0
            expected[(node - 1) * 5 + 1] = 20.0
        np.testing.assert_allclose(diagonal, expected)

    def test_project_diagonal_stiffness_identity_master(self) -> None:
        diagonal = np.array([2.0, 3.0, 5.0, 7.0])
        transformation = np.eye(4)
        master_dofs = np.arange(4)
        slave_dofs = np.array([], dtype=int)

        reduced = project_diagonal_stiffness_to_reduced(
            diagonal,
            transformation,
            master_dofs,
            slave_dofs,
        )

        np.testing.assert_allclose(reduced, np.diag(diagonal))

    def test_build_corner_mooring_reduced_stiffness_is_symmetric_positive(self) -> None:
        total_nodes = 4
        retained = 3
        transformation = np.eye(total_nodes * retained)
        master_dofs = np.arange(total_nodes * retained)
        slave_dofs = np.array([], dtype=int)

        reduced = build_corner_mooring_reduced_stiffness(
            total_nodes=total_nodes,
            retained_dofs_per_node=retained,
            nodes_per_x=2,
            nodes_per_y=2,
            transformation=transformation,
            master_dofs=master_dofs,
            slave_dofs=slave_dofs,
            surge_stiffness=1.0,
            sway_stiffness=2.0,
        )

        np.testing.assert_allclose(reduced, reduced.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(reduced) >= -1.0e-12))
        self.assertAlmostEqual(float(np.trace(reduced)), 12.0)


if __name__ == "__main__":
    unittest.main()
