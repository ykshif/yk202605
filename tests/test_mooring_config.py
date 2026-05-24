from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.mooring import (  # noqa: E402
    build_mooring_attachments_from_config,
    build_mooring_provider_from_config,
    build_reduced_mooring_terms_from_config,
    is_mooring_enabled,
    retained_full_dofs_from_case,
)


class MooringConfigTests(unittest.TestCase):
    def test_disabled_config_returns_no_provider(self) -> None:
        config = {"mooring": {"enabled": False}}

        self.assertFalse(is_mooring_enabled(config))
        self.assertIsNone(build_mooring_provider_from_config(config))
        self.assertEqual(build_mooring_attachments_from_config(config), ())

    def test_linear_matrix_attachment_accepts_diagonal_and_pretension(self) -> None:
        config = {
            "mooring": {
                "enabled": True,
                "model": "linear_matrix",
                "attachments": [
                    {
                        "name": "line_a",
                        "node_one_based": 3,
                        "stiffness": {"diagonal": [1, 2, 3, 4, 5, 6]},
                        "damping": {"diagonal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                        "pretension": [10, 20, 30, 40, 50, 60],
                    }
                ],
            }
        }

        attachment = build_mooring_attachments_from_config(config)[0]

        self.assertEqual(attachment.name, "line_a")
        self.assertEqual(attachment.node_one_based, 3)
        np.testing.assert_allclose(attachment.matrix.stiffness, np.diag([1, 2, 3, 4, 5, 6]))
        np.testing.assert_allclose(attachment.matrix.damping, np.diag([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        np.testing.assert_allclose(attachment.matrix.pretension, [10, 20, 30, 40, 50, 60])

    def test_provider_projects_configured_terms_to_reduced_coordinates(self) -> None:
        config = {
            "mooring": {
                "enabled": True,
                "model": "linear_matrix",
                "retained_full_dofs_zero_based": [0],
                "attachments": [
                    {
                        "node_one_based": 2,
                        "stiffness": {"diagonal": [4, 0, 0, 0, 0, 0]},
                        "damping": {"diagonal": [2, 0, 0, 0, 0, 0]},
                        "pretension": [8, 0, 0, 0, 0, 0],
                    }
                ],
            }
        }
        case = SimpleNamespace(
            total_nodes=2,
            full_dofs_per_node=6,
            removed_full_dofs_zero_based=(1, 2, 3, 4, 5),
        )
        structural = SimpleNamespace(
            transformation=np.array([[1.0], [0.5]]),
            master_dofs=np.array([0]),
            slave_dofs=np.array([1]),
            reverse_master_order_for_reconstruction=False,
        )

        provider = build_mooring_provider_from_config(config)
        self.assertIsNotNone(provider)
        reduced = provider(case, structural)

        np.testing.assert_allclose(reduced.stiffness, [[1.0]])
        np.testing.assert_allclose(reduced.damping, [[0.5]])
        np.testing.assert_allclose(reduced.pretension, [4.0])
        self.assertTrue(reduced.metadata["enabled"])
        self.assertEqual(reduced.metadata["attachment_count"], 1)

    def test_retained_dofs_are_inferred_from_case(self) -> None:
        case = SimpleNamespace(
            full_dofs_per_node=6,
            removed_full_dofs_zero_based=(5,),
        )

        self.assertEqual(retained_full_dofs_from_case(case), (0, 1, 2, 3, 4))

    def test_bad_matrix_shape_is_rejected(self) -> None:
        config = {
            "mooring": {
                "enabled": True,
                "attachments": [
                    {
                        "node_one_based": 1,
                        "stiffness": {"matrix": [[1.0, 2.0], [3.0, 4.0]]},
                    }
                ],
            }
        }

        with self.assertRaises(ValueError):
            build_mooring_attachments_from_config(config)


if __name__ == "__main__":
    unittest.main()
