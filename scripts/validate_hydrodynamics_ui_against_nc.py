#!/usr/bin/env python3
"""Validate the hydrodynamics UI generator against existing NetCDF baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import argparse
import json
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset  # noqa: E402


DEFAULT_UI_URL = "http://localhost:8765"
DEFAULT_LEGACY_ROOT = Path("/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoon_hinge")
RESULT_ROOT = REPO_ROOT / "results" / "hydrodynamics_ui_validation"


DOF_ORDER = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    title: str
    legacy_path: Path
    payload: dict[str, object]
    note: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-url", default=DEFAULT_UI_URL)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    require_ui(args.ui_url)

    inventory = inventory_legacy_files(args.legacy_root)
    cases = build_validation_cases(args.legacy_root)
    results = []
    for case in cases:
        print(f"[validate] running UI case: {case.case_id}")
        job = run_ui_case(args.ui_url, case.payload, timeout=args.timeout)
        generated_path = Path(job["output_path"])
        comparison = compare_static_terms(generated_path, case.legacy_path)
        consistency = check_generated_dataset_consistency(generated_path)
        results.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "legacy_path": str(case.legacy_path),
                "generated_path": str(generated_path),
                "note": case.note,
                "job_result": job.get("result", {}),
                "comparison": comparison,
                "consistency": consistency,
            }
        )

    report = {
        "ui_url": args.ui_url,
        "legacy_root": str(args.legacy_root),
        "inventory": inventory,
        "results": results,
    }
    json_path = RESULT_ROOT / "hydrodynamics_ui_validation.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = RESULT_ROOT / "hydrodynamics_ui_validation_report.md"
    report_path.write_text(render_report(report), encoding="utf-8")
    print(f"[validate] wrote {json_path}")
    print(f"[validate] wrote {report_path}")
    return 0


def require_ui(ui_url: str) -> None:
    try:
        with urlopen(ui_url + "/", timeout=5.0) as response:
            if response.status != 200:
                raise RuntimeError(f"UI returned HTTP {response.status}")
    except URLError as exc:
        raise SystemExit(
            f"Hydrodynamics UI is not reachable at {ui_url}. "
            "Start scripts/run_hydrodynamics_ui.py first."
        ) from exc


def inventory_legacy_files(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob("*.nc")):
        dataset = open_hydrodynamic_dataset(path)
        try:
            dofs = [str(value) for value in dataset.radiating_dof.values]
            bodies = sorted({dof.split("__")[0] for dof in dofs if "__" in dof})
            omega = np.atleast_1d(dataset.omega.values).astype(float)
            directions = np.atleast_1d(dataset.wave_direction.values).astype(float)
            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "sizes": {name: int(size) for name, size in dataset.sizes.items()},
                    "body_count": len(bodies),
                    "dof_count": len(dofs),
                    "rho": float(dataset.rho.values),
                    "water_depth": float(dataset.water_depth.values),
                    "omega": [float(value) for value in omega],
                    "wave_direction_deg": [float(np.rad2deg(value)) for value in directions],
                    "data_variables": list(str(name) for name in dataset.data_vars),
                    "has_required_hydro_vars": all(
                        name in dataset
                        for name in (
                            "added_mass",
                            "radiation_damping",
                            "diffraction_force",
                            "Froude_Krylov_force",
                            "inertia_matrix",
                            "hydrostatic_stiffness",
                        )
                    ),
                    "first_dofs": dofs[:6],
                }
            )
        finally:
            dataset.close()
    return rows


def build_validation_cases(legacy_root: Path) -> list[ValidationCase]:
    return [
        ValidationCase(
            case_id="dm10_slender_180_static",
            title="Yoon DM10 slender module, direction 180 deg",
            legacy_path=legacy_root / "DM10_direction180_slender180_rho1025.nc",
            payload={
                "module": {
                    "length_m": 30.0,
                    "width_m": 60.0,
                    "height_m": 4.0,
                    "draft_m": 1.1,
                    "mesh_size_m": 2.0,
                    "vertical_mesh_size_m": 0.2,
                    "mass_kg": 2029500.0,
                },
                "layout": {
                    "rows": 1,
                    "columns": 1,
                    "spacing_x_m": 30.01,
                    "spacing_y_m": 60.01,
                },
                "hydro": {
                    "rho": 1000.0,
                    "g": 9.81,
                    "n_jobs": 1,
                    "water_depth_m": None,
                    "wave_directions_deg": [180.0],
                    "omega": {"mode": "single", "single_rad_s": 0.5851},
                },
                "visual": {"wave_amplitude_m": 1.0, "motion_scale": 4.0, "speed": 1.0},
                "output_path": str(RESULT_ROOT / "ui_dm10_slender_180_static.nc"),
            },
            note=(
                "The legacy file stores rho=1025 for the BEM run, but its inertia and "
                "hydrostatic matrices match the historical notebook convention using "
                "rho=1000/static body mass. This case validates geometry, DOF labels, "
                "mass, inertia, and hydrostatics against the first module block."
            ),
        ),
        ValidationCase(
            case_id="dm10x10_square_module_static",
            title="10x10 square module, direction 0 deg",
            legacy_path=legacy_root / "DM10_10_direction0_wl180.nc",
            payload={
                "module": {
                    "length_m": 30.0,
                    "width_m": 30.0,
                    "height_m": 4.0,
                    "draft_m": 1.1,
                    "mesh_size_m": 4.0,
                    "vertical_mesh_size_m": 0.2,
                    "mass_kg": 990000.0,
                },
                "layout": {
                    "rows": 1,
                    "columns": 1,
                    "spacing_x_m": 30.01,
                    "spacing_y_m": 30.01,
                },
                "hydro": {
                    "rho": 1000.0,
                    "g": 9.81,
                    "n_jobs": 1,
                    "water_depth_m": None,
                    "wave_directions_deg": [0.0],
                    "omega": {"mode": "single", "single_rad_s": 0.5851},
                },
                "visual": {"wave_amplitude_m": 1.0, "motion_scale": 4.0, "speed": 1.0},
                "output_path": str(RESULT_ROOT / "ui_dm10x10_square_module_static.nc"),
            },
            note=(
                "This is a 1-module UI calculation compared with the first module "
                "static block of the 100-module legacy array. Radiation/excitation "
                "terms are intentionally not compared because the legacy file includes "
                "multi-body hydrodynamic interactions."
            ),
        ),
    ]


def run_ui_case(ui_url: str, payload: dict[str, object], *, timeout: float) -> dict[str, object]:
    request = Request(
        ui_url + "/api/run",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10.0) as response:
        job = json.loads(response.read().decode("utf-8"))
    job_id = job["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with urlopen(ui_url + f"/api/jobs/{job_id}", timeout=10.0) as response:
            status = json.loads(response.read().decode("utf-8"))
        if status["status"] in {"completed", "failed"}:
            if status["status"] != "completed":
                raise RuntimeError(f"UI job {job_id} failed:\n" + "\n".join(status.get("logs", [])))
            return status
        time.sleep(0.75)
    raise TimeoutError(f"UI job {job_id} did not finish within {timeout} seconds")


def compare_static_terms(generated_path: Path, legacy_path: Path) -> dict[str, object]:
    generated = open_hydrodynamic_dataset(generated_path)
    legacy = open_hydrodynamic_dataset(legacy_path)
    try:
        generated_labels = [str(label) for label in generated.radiating_dof.values[:6]]
        legacy_labels = [str(label) for label in legacy.radiating_dof.values[:6]]
        expected = [f"0_0__{dof}" for dof in DOF_ORDER]
        metrics = {
            "dof_labels_match_expected": generated_labels == expected and legacy_labels == expected,
            "generated_first_dofs": generated_labels,
            "legacy_first_dofs": legacy_labels,
        }
        for variable in ("inertia_matrix", "hydrostatic_stiffness"):
            generated_block = select_block(generated[variable], generated_labels).values
            legacy_block = select_block(legacy[variable], legacy_labels).values
            metrics[variable] = matrix_error_metrics(generated_block, legacy_block)
        return metrics
    finally:
        generated.close()
        legacy.close()


def select_block(data_array, labels: list[str]):
    return data_array.sel(influenced_dof=labels, radiating_dof=labels)


def matrix_error_metrics(generated: np.ndarray, legacy: np.ndarray) -> dict[str, object]:
    difference = generated - legacy
    scale = np.maximum(1.0, np.abs(legacy))
    diagonal_generated = np.diag(generated)
    diagonal_legacy = np.diag(legacy)
    diagonal_scale = np.maximum(1.0, np.abs(diagonal_legacy))
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "max_relative_error": float(np.max(np.abs(difference) / scale)),
        "l2_relative_error": float(np.linalg.norm(difference) / max(1.0, np.linalg.norm(legacy))),
        "diagonal_generated": [float(value) for value in diagonal_generated],
        "diagonal_legacy": [float(value) for value in diagonal_legacy],
        "diagonal_max_relative_error": float(
            np.max(np.abs(diagonal_generated - diagonal_legacy) / diagonal_scale)
        ),
    }


def check_generated_dataset_consistency(path: Path) -> dict[str, object]:
    dataset = open_hydrodynamic_dataset(path)
    try:
        has_excitation = "excitation_force" in dataset
        excitation_error = None
        if has_excitation:
            expected = dataset["Froude_Krylov_force"] + dataset["diffraction_force"]
            excitation_error = float(np.max(np.abs(dataset["excitation_force"].values - expected.values)))
        return {
            "sizes": {name: int(size) for name, size in dataset.sizes.items()},
            "data_variables": list(str(name) for name in dataset.data_vars),
            "has_rao": "rao" in dataset,
            "has_excitation_force": has_excitation,
            "excitation_force_identity_max_abs_error": excitation_error,
        }
    finally:
        dataset.close()


def render_report(report: dict[str, object]) -> str:
    lines = [
        "# Hydrodynamics UI NetCDF Validation",
        "",
        f"- UI URL: `{report['ui_url']}`",
        f"- Legacy root: `{report['legacy_root']}`",
        "",
        "## Legacy NetCDF Inventory",
        "",
        "| file | bodies | DOF | omega | wave dir deg | rho | required vars |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in report["inventory"]:
        lines.append(
            "| {file} | {body_count} | {dof_count} | {omega} | {direction} | {rho:g} | {ok} |".format(
                file=row["file"],
                body_count=row["body_count"],
                dof_count=row["dof_count"],
                omega=", ".join(f"{v:.8g}" for v in row["omega"]),
                direction=", ".join(f"{v:.8g}" for v in row["wave_direction_deg"]),
                rho=row["rho"],
                ok="yes" if row["has_required_hydro_vars"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## UI Recalculation Comparisons",
            "",
            "| case | generated file | inertia max rel | hydrostatic max rel | RAO | excitation identity |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for result in report["results"]:
        comparison = result["comparison"]
        consistency = result["consistency"]
        lines.append(
            "| {case} | `{file}` | {im:.6g} | {hs:.6g} | {rao} | {ex} |".format(
                case=result["case_id"],
                file=result["generated_path"],
                im=comparison["inertia_matrix"]["max_relative_error"],
                hs=comparison["hydrostatic_stiffness"]["max_relative_error"],
                rao="yes" if consistency["has_rao"] else "no",
                ex=(
                    consistency["excitation_force_identity_max_abs_error"]
                    if consistency["excitation_force_identity_max_abs_error"] is not None
                    else float("nan")
                ),
            )
        )

    lines.extend(["", "## Detailed Notes", ""])
    for result in report["results"]:
        comparison = result["comparison"]
        consistency = result["consistency"]
        lines.extend(
            [
                f"### {result['case_id']}",
                "",
                result["note"],
                "",
                f"- Legacy: `{result['legacy_path']}`",
                f"- Generated through UI API: `{result['generated_path']}`",
                f"- DOF labels match expected `0_0__Surge ... 0_0__Yaw`: `{comparison['dof_labels_match_expected']}`",
                f"- Generated variables: `{', '.join(consistency['data_variables'])}`",
                f"- Inertia diagonal generated: `{format_float_list(comparison['inertia_matrix']['diagonal_generated'])}`",
                f"- Inertia diagonal legacy: `{format_float_list(comparison['inertia_matrix']['diagonal_legacy'])}`",
                f"- Hydrostatic diagonal generated: `{format_float_list(comparison['hydrostatic_stiffness']['diagonal_generated'])}`",
                f"- Hydrostatic diagonal legacy: `{format_float_list(comparison['hydrostatic_stiffness']['diagonal_legacy'])}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            "The UI-generated NetCDF files use the same Capytaine-style variable names, "
            "six-DOF body labels, separated complex-value convention, and add a new `rao` "
            "variable for motion visualization. The strict numerical comparisons are made "
            "on inertia and hydrostatic blocks because these are single-body properties and "
            "can be compared directly with the first-module blocks of the legacy array files. "
            "Radiation, diffraction, and Froude-Krylov terms from a one-module UI run are not "
            "expected to match a multi-body legacy array because those legacy terms include "
            "hydrodynamic interactions between modules.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float_list(values: list[float]) -> str:
    return ", ".join(f"{value:.8g}" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
