from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    MooringLinearization,
    WecSimLikeRadiationConfig,
)
from offshore_energy_sim.time_domain_adapter.wecsim_like_solver import _resolve_mooring  # noqa: E402


class WecSimLikeAdapterTests(unittest.TestCase):
    def test_radiation_config_validation(self) -> None:
        self.assertEqual(WecSimLikeRadiationConfig(model="STATE_SPACE").model, "state_space")
        self.assertEqual(WecSimLikeRadiationConfig(integrator="RK4").integrator, "rk4")
        with self.assertRaises(ValueError):
            WecSimLikeRadiationConfig(model="bad")
        with self.assertRaises(ValueError):
            WecSimLikeRadiationConfig(model="state_space", state_space_order=0)
        with self.assertRaises(ValueError):
            WecSimLikeRadiationConfig(integrator="bad")

    def test_resolve_mooring_accepts_reduced_linearization(self) -> None:
        stiffness = np.array([[2.0, 0.1], [0.0, 3.0]])
        damping = np.array([[0.2, 0.02], [0.0, 0.3]])
        pretension = np.array([5.0, -1.0])
        mooring = MooringLinearization(
            stiffness,
            metadata={"kind": "unit_test"},
            reduced_damping=damping,
            reduced_pretension=pretension,
        )

        resolved = _resolve_mooring(mooring, None, None, None, 2)

        np.testing.assert_allclose(resolved.reduced_stiffness, [[2.0, 0.05], [0.05, 3.0]])
        np.testing.assert_allclose(resolved.reduced_damping, [[0.2, 0.01], [0.01, 0.3]])
        np.testing.assert_allclose(resolved.reduced_pretension, pretension)
        self.assertTrue(resolved.metadata["enabled"])
        self.assertEqual(resolved.metadata["kind"], "unit_test")

    def test_resolve_mooring_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_mooring(np.eye(3), None, None, None, 2)


if __name__ == "__main__":
    unittest.main()
