from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.time_domain_adapter import (  # noqa: E402
    build_radiation_kernel,
    compare_radiation_kernels,
    radiation_kernel_diagnostics,
)


class RadiationKernelAdapterTests(unittest.TestCase):
    def test_build_radiation_kernel_matches_expected_shape(self) -> None:
        omega = np.linspace(0.1, 3.0, 40)
        time = np.linspace(0.0, 8.0, 81)
        damping = np.exp(-omega)[:, np.newaxis, np.newaxis] * np.array([[[2.0]]])

        kernel = build_radiation_kernel(omega, damping, time)

        self.assertEqual(kernel.shape, (time.size, 1, 1))
        self.assertTrue(np.all(np.isfinite(kernel)))

    def test_radiation_kernel_diagnostics_identifies_decaying_tail(self) -> None:
        time = np.linspace(0.0, 10.0, 101)
        kernel = np.exp(-time)[:, np.newaxis, np.newaxis] * np.array([[[3.0]]])

        diagnostics = radiation_kernel_diagnostics(time, kernel)

        self.assertLess(diagnostics.tail_rms_to_peak_ratio, 1.0e-2)
        self.assertEqual(diagnostics.dof_count, 1)

    def test_compare_radiation_kernels_reports_tail_improvement(self) -> None:
        time = np.linspace(0.0, 10.0, 101)
        before = np.exp(-0.2 * time)[:, np.newaxis, np.newaxis]
        after = np.exp(-time)[:, np.newaxis, np.newaxis]

        comparison = compare_radiation_kernels(time, before, after)

        self.assertLess(comparison["tail_rms_ratio_after_over_before"], 1.0)
        self.assertIn("before", comparison)
        self.assertIn("after", comparison)


if __name__ == "__main__":
    unittest.main()
