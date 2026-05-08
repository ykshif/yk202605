from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.strength import (  # noqa: E402
    Connector,
    assemble_connector_dynamic_stiffness,
    build_case_hinge_pair_connectors,
    build_direct_relative_G,
    build_hinge_pair_connectors,
    build_weighted_endpoint_operator,
    build_weighted_relative_G,
    connector_force_envelope,
    harmonic_vector_norm_envelope,
    recover_connector_response,
)
from offshore_energy_sim.structure import ExplicitHingeSpec  # noqa: E402


class ConnectorRecoveryTests(unittest.TestCase):
    def test_two_node_three_dof_spring_force(self) -> None:
        G = build_direct_relative_G(
            ndof=6,
            dofs_a=(0, 1, 2),
            dofs_b=(3, 4, 5),
        )
        connector = Connector("spring", G, K=np.array([1000.0, 1000.0, 1000.0]))
        x_hat = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

        recovered = recover_connector_response(x_hat, omega=1.0, connectors=(connector,))

        np.testing.assert_allclose(recovered["spring"]["delta_hat"], [0.01, 0.0, 0.0])
        np.testing.assert_allclose(recovered["spring"]["force_hat"], [10.0, 0.0, 0.0])

    def test_same_endpoint_motion_gives_zero_delta_and_force(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector("spring", G, K=np.array([1000.0, 1000.0, 1000.0]))
        x_hat = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])

        recovered = recover_connector_response(x_hat, omega=2.0, connectors=(connector,))

        np.testing.assert_allclose(recovered["spring"]["delta_hat"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(recovered["spring"]["force_hat"], [0.0, 0.0, 0.0])

    def test_zero_stiffness_gives_zero_force(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector("free", G, K=np.zeros(3))
        x_hat = np.array([0.01, 0.02, 0.03, 0.0, 0.0, 0.0])

        recovered = recover_connector_response(x_hat, omega=1.0, connectors=(connector,))

        np.testing.assert_allclose(recovered["free"]["force_hat"], [0.0, 0.0, 0.0])

    def test_assembled_dynamic_stiffness_matches_recovered_global_force(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector("spring", G, K=np.array([1000.0, 2000.0, 3000.0]))
        x_hat = np.array([0.01, 0.02, -0.03, 0.001, -0.002, 0.003])

        recovered = recover_connector_response(x_hat, omega=1.0, connectors=(connector,))
        force_hat = recovered["spring"]["force_hat"]
        Zc = assemble_connector_dynamic_stiffness(6, (connector,), omega=1.0)

        np.testing.assert_allclose(G.T @ force_hat, Zc @ x_hat)

    def test_complex_x_hat_is_preserved(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector("complex", G, K=np.array([1000.0, 1000.0, 1000.0]))
        x_hat = np.array([0.01 + 0.02j, 0.0, 0.0, 0.0, 0.0, 0.0])

        recovered = recover_connector_response(x_hat, omega=1.0, connectors=(connector,))

        np.testing.assert_allclose(recovered["complex"]["delta_hat"], [0.01 + 0.02j, 0.0, 0.0])
        np.testing.assert_allclose(recovered["complex"]["force_hat"], [10.0 + 20.0j, 0.0, 0.0])

    def test_multiple_frequency_cases_with_damping(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector(
            "damped",
            G,
            K=np.array([1000.0, 1000.0, 1000.0]),
            C=np.array([10.0, 10.0, 10.0]),
        )
        x_hat = np.array(
            [
                [0.01, 0.02],
                [0.00, 0.00],
                [0.00, 0.00],
                [0.00, 0.00],
                [0.00, 0.00],
                [0.00, 0.00],
            ],
            dtype=np.complex128,
        )

        recovered = recover_connector_response(
            x_hat,
            omega=np.array([1.0, 2.0]),
            connectors=(connector,),
        )

        expected = np.array(
            [
                [(1000.0 + 1j * 1.0 * 10.0) * 0.01, (1000.0 + 1j * 2.0 * 10.0) * 0.02],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(recovered["damped"]["force_hat"], expected)

    def test_weighted_endpoint_and_local_transform(self) -> None:
        dofmap = {
            1: {"ux": 0, "uy": 1, "uz": 2},
            2: {"ux": 3, "uy": 4, "uz": 5},
            3: {"ux": 6, "uy": 7, "uz": 8},
            4: {"ux": 9, "uy": 10, "uz": 11},
        }
        H_a = build_weighted_endpoint_operator(12, {1: 0.25, 2: 0.75}, dofmap, ("ux", "uy", "uz"))
        H_b = build_weighted_endpoint_operator(12, {3: 1.0}, dofmap, ("ux", "uy", "uz"))
        R = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        G = build_weighted_relative_G(H_a, H_b, R=R, select=(0, 2))
        x_hat = np.arange(12.0)

        raw = H_a @ x_hat - H_b @ x_hat
        expected = (R @ raw)[[0, 2]]
        np.testing.assert_allclose(G @ x_hat, expected)

    def test_harmonic_envelope_uses_phase_not_direct_real_part(self) -> None:
        envelope, angle = harmonic_vector_norm_envelope(np.array([1.0 + 1.0j]))

        np.testing.assert_allclose(envelope, np.sqrt(2.0))
        self.assertTrue(-np.pi <= angle <= np.pi)

    def test_connector_force_envelope_accepts_recovered_mapping(self) -> None:
        G = build_direct_relative_G(6, (0, 1, 2), (3, 4, 5))
        connector = Connector("spring", G, K=np.ones(3))
        x_hat = np.array([1.0 + 1.0j, 0.0, 0.0, 0.0, 0.0, 0.0])
        recovered = recover_connector_response(x_hat, omega=1.0, connectors=(connector,))

        envelopes = connector_force_envelope(recovered)

        np.testing.assert_allclose(envelopes["spring"]["envelope"], np.sqrt(2.0))

    def test_hinge_specs_convert_to_pair_connectors(self) -> None:
        hinge = ExplicitHingeSpec(
            nodes_side_a_one_based=(1, 2),
            nodes_side_b_one_based=(3, 4),
            k_hinge=10.0,
            dofs_per_node=6,
            released_dofs_zero_based=(4,),
            released_dof_stiffness=2.0,
            name="test hinge",
        )
        connectors = build_hinge_pair_connectors(
            (hinge,),
            total_nodes=4,
            response_dofs_per_node=5,
            removed_full_dofs_zero_based=(5,),
        )

        class FakeCase:
            total_nodes = 4
            hinges = (hinge,)
            retained_dofs_per_node = 5
            removed_full_dofs_zero_based = (5,)

        case_connectors = build_case_hinge_pair_connectors(FakeCase())
        x_hat = np.arange(20, dtype=float) + 1j * np.arange(20, dtype=float) / 10.0

        recovered = recover_connector_response(x_hat, omega=0.0, connectors=connectors)

        self.assertEqual(len(connectors), 2)
        self.assertEqual(len(case_connectors), 2)
        first = recovered[connectors[0].cid]
        expected_delta = x_hat[0:5] - x_hat[10:15]
        expected_force = np.array([10.0, 10.0, 10.0, 10.0, 2.0]) * expected_delta
        np.testing.assert_allclose(first["delta_hat"], expected_delta)
        np.testing.assert_allclose(first["force_hat"], expected_force)
        self.assertEqual(connectors[0].meta["released_labels"], ("ry",))


if __name__ == "__main__":
    unittest.main()
