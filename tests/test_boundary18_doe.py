from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.optimization import (  # noqa: E402
    BOUNDARY18_GROUP_NAMES,
    Boundary18Sample,
    generate_boundary18_doe_samples,
    generate_boundary18_refined_samples,
)


class Boundary18DoeTests(unittest.TestCase):
    def test_standard_group_names_have_18_entries(self) -> None:
        self.assertEqual(len(BOUNDARY18_GROUP_NAMES), 18)
        self.assertEqual(BOUNDARY18_GROUP_NAMES[0], "x_boundary_01")
        self.assertEqual(BOUNDARY18_GROUP_NAMES[8], "x_boundary_09")
        self.assertEqual(BOUNDARY18_GROUP_NAMES[9], "y_boundary_01")
        self.assertEqual(BOUNDARY18_GROUP_NAMES[-1], "y_boundary_09")

    def test_generate_doe_samples_are_deterministic_and_bounded(self) -> None:
        samples_a = generate_boundary18_doe_samples(low=1.0e8, high=1.0e9, random_count=2, seed=7)
        samples_b = generate_boundary18_doe_samples(low=1.0e8, high=1.0e9, random_count=2, seed=7)

        self.assertEqual(len(samples_a), 12)
        self.assertEqual([sample.name for sample in samples_a], [sample.name for sample in samples_b])
        for sample_a, sample_b in zip(samples_a, samples_b):
            np.testing.assert_allclose(sample_a.values, sample_b.values)
            self.assertEqual(len(sample_a.values), 18)
            self.assertGreaterEqual(min(sample_a.values), 1.0e8)
            self.assertLessEqual(max(sample_a.values), 1.0e9)

    def test_sample_value_by_group(self) -> None:
        sample = Boundary18Sample("unit", tuple(float(index) for index in range(18)))
        mapping = sample.value_by_group()

        self.assertEqual(mapping["x_boundary_01"], 0.0)
        self.assertEqual(mapping["y_boundary_09"], 17.0)

    def test_generate_refined_samples_cover_promising_regions(self) -> None:
        samples = generate_boundary18_refined_samples(low=1.0e8, high=1.0e9)
        names = [sample.name for sample in samples]

        self.assertEqual(len(samples), 18)
        self.assertIn("anchor_x_high_y_low", names)
        self.assertIn("anchor_center_stiff", names)
        self.assertIn("anchor_uniform_mid", names)
        for sample in samples:
            self.assertEqual(len(sample.values), 18)
            self.assertGreaterEqual(min(sample.values), 1.0e8)
            self.assertLessEqual(max(sample.values), 1.0e9)


if __name__ == "__main__":
    unittest.main()
