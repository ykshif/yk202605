from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    build_array_body,
    module_structural_node_mappings,
    omega_from_wavelength,
    omega_values_from_range,
    omega_values_from_wavelengths,
    parse_float_sequence,
    preview_layout,
    summarize_rao_for_ui,
)


class CapytaineArrayTests(unittest.TestCase):
    def test_omega_range_and_sequence_parsing(self) -> None:
        np.testing.assert_allclose(
            omega_values_from_range(0.1, 0.3, 3),
            (0.1, 0.2, 0.3),
        )
        self.assertEqual(parse_float_sequence("0.4, 0.5; 0.6\n0.7"), (0.4, 0.5, 0.6, 0.7))

    def test_wavelength_to_omega_matches_linear_dispersion_relation(self) -> None:
        wavelength = 300.0
        water_depth = 58.5
        g = 9.81
        k = 2.0 * np.pi / wavelength

        omega = omega_from_wavelength(wavelength, water_depth, g)

        np.testing.assert_allclose(omega, np.sqrt(g * k * np.tanh(k * water_depth)))
        np.testing.assert_allclose(
            omega_values_from_wavelengths("180, 300", water_depth, g),
            (
                np.sqrt(g * (2.0 * np.pi / 180.0) * np.tanh((2.0 * np.pi / 180.0) * water_depth)),
                omega,
            ),
        )

    def test_mesh_resolution_matches_legacy_notebook_flooring(self) -> None:
        module = RectangularModuleSpec(30.0, 30.0, 4.0, 1.1, 4.0, vertical_mesh_size_m=0.2)

        self.assertEqual(module.mesh_resolution, (7, 7, 20))

    def test_layout_preview_counts_bem_problems(self) -> None:
        config = ArrayHydrodynamicsConfig(
            module=RectangularModuleSpec(2.0, 1.0, 1.0, 0.5, 1.0),
            layout=ArrayLayoutSpec(rows=2, columns=3, spacing_x_m=2.0, spacing_y_m=1.0),
            omegas_rad_s=(0.5, 0.7),
            wave_directions_rad=(0.0, np.pi),
            output_path=Path("dummy.nc"),
        )

        preview = preview_layout(config)

        self.assertEqual(preview["body_count"], 6)
        self.assertEqual(preview["dof_count"], 36)
        self.assertEqual(preview["radiation_problem_count"], 72)
        self.assertEqual(preview["diffraction_problem_count"], 4)
        self.assertEqual(preview["centers"][0]["name"], "0_0")
        self.assertEqual(preview["centers"][-1]["name"], "2_1")

    def test_custom_1d_module_lengths_define_boundaries_and_centers(self) -> None:
        lengths = (15.0, 45.0, 30.0, 30.0, 20.0, 40.0, 25.0, 35.0, 30.0, 30.0)
        layout = ArrayLayoutSpec(
            rows=1,
            columns=10,
            spacing_x_m=30.0,
            spacing_y_m=60.0,
            division_mode="custom",
            total_length_m=300.0,
            module_lengths_x_m=lengths,
        )
        config = ArrayHydrodynamicsConfig(
            module=RectangularModuleSpec(30.0, 60.0, 2.0, 1.0, 5.0),
            layout=layout,
            omegas_rad_s=(0.5,),
            output_path=Path("dummy.nc"),
        )

        preview = preview_layout(config)

        self.assertEqual(preview["body_count"], 10)
        self.assertEqual(preview["division_mode"], "custom")
        self.assertEqual(preview["module_lengths_x_m"], list(lengths))
        np.testing.assert_allclose(
            preview["x_boundaries_m"],
            [0.0, 15.0, 60.0, 90.0, 120.0, 140.0, 180.0, 205.0, 240.0, 270.0, 300.0],
        )
        np.testing.assert_allclose(
            [item["x_m"] for item in preview["centers"]],
            [7.5, 37.5, 75.0, 105.0, 130.0, 160.0, 192.5, 222.5, 255.0, 285.0],
        )
        self.assertTrue(all(item["length_m"] > 0 for item in preview["centers"]))
        self.assertTrue(all(item["y_m"] == 0.0 for item in preview["centers"]))

    def test_custom_1d_module_lengths_validate_count_and_sum(self) -> None:
        with self.assertRaises(ValueError):
            ArrayLayoutSpec(
                rows=1,
                columns=10,
                spacing_x_m=30.0,
                spacing_y_m=60.0,
                division_mode="custom",
                total_length_m=300.0,
                module_lengths_x_m=(30.0, 30.0),
            )

        with self.assertRaises(ValueError):
            ArrayLayoutSpec(
                rows=1,
                columns=3,
                spacing_x_m=30.0,
                spacing_y_m=60.0,
                division_mode="custom",
                total_length_m=300.0,
                module_lengths_x_m=(100.0, 100.0, 50.0),
            )

        with self.assertRaises(ValueError):
            ArrayLayoutSpec(
                rows=2,
                columns=3,
                spacing_x_m=30.0,
                spacing_y_m=60.0,
                division_mode="custom",
                total_length_m=300.0,
                module_lengths_x_m=(100.0, 100.0, 100.0),
            )

    def test_structural_grid_maps_module_centers_to_793_node_mesh(self) -> None:
        lengths = (20.0, 40.0, 30.0, 30.0, 20.0, 40.0, 20.0, 40.0, 30.0, 30.0)
        config = ArrayHydrodynamicsConfig(
            module=RectangularModuleSpec(30.0, 60.0, 2.0, 1.0, 5.0),
            layout=ArrayLayoutSpec(
                rows=1,
                columns=10,
                spacing_x_m=30.0,
                spacing_y_m=60.0,
                division_mode="custom",
                total_length_m=300.0,
                module_lengths_x_m=lengths,
            ),
            omegas_rad_s=(0.5,),
            output_path=Path("dummy.nc"),
            structural_grid=StructuralGridSpec(length_m=300.0, width_m=60.0, dx_m=5.0, dy_m=5.0),
        )

        mappings = module_structural_node_mappings(config)
        preview = preview_layout(config)

        self.assertEqual(config.structural_grid.total_nodes, 793)
        self.assertEqual([item["x_m"] for item in mappings], [10.0, 40.0, 75.0, 105.0, 130.0, 160.0, 190.0, 220.0, 255.0, 285.0])
        self.assertEqual([item["fem_node_one_based"] for item in mappings], [425, 419, 412, 406, 401, 395, 389, 383, 376, 370])
        self.assertEqual(preview["structural_grid"]["total_nodes"], 793)
        self.assertEqual(
            [item["fem_node_one_based"] for item in preview["structural_nodes"]],
            [425, 419, 412, 406, 401, 395, 389, 383, 376, 370],
        )

    def test_structural_grid_rejects_off_node_module_centers(self) -> None:
        lengths = (15.0, 45.0, 30.0, 30.0, 20.0, 40.0, 25.0, 35.0, 30.0, 30.0)
        with self.assertRaises(ValueError):
            ArrayHydrodynamicsConfig(
                module=RectangularModuleSpec(30.0, 60.0, 2.0, 1.0, 5.0),
                layout=ArrayLayoutSpec(
                    rows=1,
                    columns=10,
                    spacing_x_m=30.0,
                    spacing_y_m=60.0,
                    division_mode="custom",
                    total_length_m=300.0,
                    module_lengths_x_m=lengths,
                ),
                omegas_rad_s=(0.5,),
                output_path=Path("dummy.nc"),
                structural_grid=StructuralGridSpec(length_m=300.0, width_m=60.0, dx_m=5.0, dy_m=5.0),
            )

    def test_summarize_rao_for_ui_groups_body_dofs(self) -> None:
        rao = xr.DataArray(
            np.array([[[1.0 + 2.0j, 0.5 - 0.5j]]]),
            coords={
                "omega": [0.5],
                "wave_direction": [0.0],
                "radiating_dof": ["0_0__Heave", "0_0__Pitch"],
            },
            dims=("omega", "wave_direction", "radiating_dof"),
        )

        summary = summarize_rao_for_ui(rao)

        self.assertEqual(summary["body_count"], 1)
        self.assertEqual(summary["bodies"][0]["name"], "0_0")
        np.testing.assert_allclose(summary["bodies"][0]["dofs"]["Heave"]["abs"], np.sqrt(5.0))
        np.testing.assert_allclose(summary["bodies"][0]["dofs"]["Pitch"]["phase_rad"], -np.pi / 4)

    @unittest.skipUnless(find_spec("capytaine") is not None, "Capytaine is not installed")
    def test_build_array_body_uses_rodm_dof_labels(self) -> None:
        config = ArrayHydrodynamicsConfig(
            module=RectangularModuleSpec(2.0, 1.0, 1.0, 0.5, 2.0),
            layout=ArrayLayoutSpec(rows=1, columns=2, spacing_x_m=2.0, spacing_y_m=1.0),
            omegas_rad_s=(0.5,),
            output_path=Path("dummy.nc"),
        )

        body = build_array_body(config)

        self.assertEqual(
            list(body.dofs.keys())[:6],
            [
                "0_0__Surge",
                "0_0__Sway",
                "0_0__Heave",
                "0_0__Roll",
                "0_0__Pitch",
                "0_0__Yaw",
            ],
        )
        self.assertEqual(len(body.dofs), 12)
        self.assertEqual(body.inertia_matrix.shape, (12, 12))
        self.assertEqual(body.hydrostatic_stiffness.shape, (12, 12))


if __name__ == "__main__":
    unittest.main()
