from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.mooring import (  # noqa: E402
    LinearMooringMatrix,
    NodalMooringAttachment,
    assemble_nodal_mooring_terms,
    build_nodal_mooring_reduced_terms,
    project_global_mooring_terms_to_reduced,
)


class MooringMatrixTests(unittest.TestCase):
    def test_linear_mooring_force_uses_wecsim_matrix_convention(self) -> None:
        mooring = LinearMooringMatrix(
            stiffness=np.diag([2.0, 3.0]),
            damping=np.diag([0.5, 0.25]),
            pretension=np.array([10.0, -4.0]),
            dof_count=2,
        )

        force = mooring.force(
            np.array([1.0, 2.0]),
            np.array([4.0, -8.0]),
        )

        np.testing.assert_allclose(force, [6.0, -8.0])

    def test_nodal_assembly_drops_removed_yaw_dof(self) -> None:
        stiffness = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 99.0])
        damping = np.diag([0.1, 0.2, 0.3, 0.4, 0.5, 9.9])
        pretension = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 990.0])
        attachment = NodalMooringAttachment(
            node_one_based=2,
            matrix=LinearMooringMatrix(
                stiffness=stiffness,
                damping=damping,
                pretension=pretension,
            ),
            name="stern_line",
        )

        terms = assemble_nodal_mooring_terms(
            [attachment],
            total_nodes=2,
            retained_full_dofs_zero_based=(0, 1, 2, 3, 4),
        )

        expected_stiffness = np.zeros((10, 10))
        expected_stiffness[5:10, 5:10] = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_damping = np.zeros((10, 10))
        expected_damping[5:10, 5:10] = np.diag([0.1, 0.2, 0.3, 0.4, 0.5])
        expected_pretension = np.zeros(10)
        expected_pretension[5:10] = [10.0, 20.0, 30.0, 40.0, 50.0]

        np.testing.assert_allclose(terms.stiffness, expected_stiffness)
        np.testing.assert_allclose(terms.damping, expected_damping)
        np.testing.assert_allclose(terms.pretension, expected_pretension)
        self.assertEqual(terms.metadata["attachment_names"], ("stern_line",))

    def test_project_global_terms_to_reduced_identity_transform(self) -> None:
        attachment = NodalMooringAttachment(
            node_one_based=1,
            matrix=LinearMooringMatrix(
                stiffness=np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
                damping=np.diag([0.2, 0.3, 0.5, 0.7, 1.1, 1.3]),
                pretension=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
        )
        terms = assemble_nodal_mooring_terms(
            [attachment],
            total_nodes=1,
            retained_full_dofs_zero_based=(0, 1, 2, 3, 4),
        )

        reduced = project_global_mooring_terms_to_reduced(
            terms,
            np.eye(5),
            np.arange(5),
            np.array([], dtype=int),
        )

        np.testing.assert_allclose(reduced.stiffness, terms.stiffness)
        np.testing.assert_allclose(reduced.damping, terms.damping)
        np.testing.assert_allclose(reduced.pretension, terms.pretension)
        self.assertTrue(reduced.enabled)

    def test_build_nodal_reduced_terms_projects_slave_dof_coupling(self) -> None:
        attachment = NodalMooringAttachment(
            node_one_based=2,
            matrix=LinearMooringMatrix(
                stiffness=np.diag([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ),
        )
        # Two retained nodes with one retained DOF each. The slave DOF equals
        # 0.5 times the master coordinate, so K_reduced = 4 * 0.5**2.
        transformation = np.array([[1.0], [0.5]])

        reduced = build_nodal_mooring_reduced_terms(
            [attachment],
            total_nodes=2,
            retained_full_dofs_zero_based=(0,),
            transformation=transformation,
            master_dofs=np.array([0]),
            slave_dofs=np.array([1]),
        )

        np.testing.assert_allclose(reduced.stiffness, [[1.0]])


if __name__ == "__main__":
    unittest.main()
