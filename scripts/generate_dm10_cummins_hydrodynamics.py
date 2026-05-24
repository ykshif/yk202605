"""Generate a multi-frequency 10-module hydrodynamic dataset for Cummins runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    ArrayHydrodynamicsConfig,
    ArrayLayoutSpec,
    RectangularModuleSpec,
    StructuralGridSpec,
    parse_float_sequence,
    preview_layout,
    run_array_hydrodynamics,
)
from offshore_energy_sim.utils.hashing import sha256_file  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "external"
    / "DM-FEM2D"
    / "HydrodynamicData"
    / "Yoga"
    / "DM10_direction0_cummins_omega0p10_2p00_9plus_target_mesh4.nc"
)
BENCHMARK_OMEGA = 0.4157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing NetCDF file.")
    parser.add_argument("--omega-start", type=float, default=0.1)
    parser.add_argument("--omega-stop", type=float, default=2.0)
    parser.add_argument("--omega-count", type=int, default=9)
    parser.add_argument(
        "--omega-values",
        type=str,
        default=None,
        help=(
            "Comma/space separated custom angular-frequency grid. When set, "
            "--omega-start/--omega-stop/--omega-count are ignored."
        ),
    )
    parser.add_argument(
        "--omega-file",
        type=Path,
        default=None,
        help=(
            "Text file containing custom angular frequencies separated by "
            "commas, whitespace, or newlines. Overrides --omega-values."
        ),
    )
    parser.add_argument(
        "--include-omega",
        type=float,
        action="append",
        default=[BENCHMARK_OMEGA],
        help="Additional frequency to insert exactly; repeat for multiple values.",
    )
    parser.add_argument("--wave-direction-deg", type=float, default=0.0)
    parser.add_argument("--total-length", type=float, default=300.0)
    parser.add_argument("--module-count", type=int, default=10)
    parser.add_argument("--module-width", type=float, default=60.0)
    parser.add_argument("--module-height", type=float, default=2.0)
    parser.add_argument("--draft", type=float, default=0.5)
    parser.add_argument("--mesh-size", type=float, default=4.0)
    parser.add_argument("--vertical-mesh-size", type=float, default=2.0)
    parser.add_argument("--water-depth", type=float, default=58.5)
    parser.add_argument("--rho", type=float, default=1025.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument(
        "--static-rho",
        type=float,
        default=1000.0,
        help=(
            "Scale hydrostatic stiffness after the BEM solve to match the legacy "
            "Yoga static convention. Use 0 to keep Capytaine's hydrostatic matrix."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) - 2)),
    )
    parser.add_argument(
        "--compute-rao",
        action="store_true",
        help="Also ask Capytaine to compute RAOs. Usually off for Cummins preprocessing.",
    )
    return parser.parse_args()


def omega_grid(args: argparse.Namespace) -> tuple[float, ...]:
    if args.omega_file is not None:
        values = np.array(parse_float_sequence(args.omega_file.read_text(encoding="utf-8")), dtype=float)
    elif args.omega_values is not None:
        values = np.array(parse_float_sequence(args.omega_values), dtype=float)
    else:
        values = np.linspace(args.omega_start, args.omega_stop, args.omega_count)
        extra = np.array(args.include_omega or [], dtype=float)
        values = np.concatenate([values, extra])
    values = values[(values > 0.0) & np.isfinite(values)]
    rounded = sorted({round(float(value), 12) for value in values})
    return tuple(float(value) for value in rounded)


def build_config(args: argparse.Namespace) -> ArrayHydrodynamicsConfig:
    module_length = args.total_length / args.module_count
    module = RectangularModuleSpec(
        length_m=module_length,
        width_m=args.module_width,
        height_m=args.module_height,
        draft_m=args.draft,
        mesh_size_m=args.mesh_size,
        vertical_mesh_size_m=args.vertical_mesh_size,
        mass_kg=module_length * args.module_width * args.draft * args.rho,
    )
    layout = ArrayLayoutSpec(
        rows=1,
        columns=args.module_count,
        spacing_x_m=module_length,
        spacing_y_m=args.module_width,
        division_mode="uniform",
        total_length_m=args.total_length,
    )
    return ArrayHydrodynamicsConfig(
        module=module,
        layout=layout,
        omegas_rad_s=omega_grid(args),
        output_path=args.output,
        wave_directions_rad=(float(np.deg2rad(args.wave_direction_deg)),),
        water_depth_m=args.water_depth,
        rho=args.rho,
        g=args.g,
        n_jobs=args.n_jobs,
        compute_rao=args.compute_rao,
        structural_grid=StructuralGridSpec(
            length_m=args.total_length,
            width_m=args.module_width,
            dx_m=5.0,
            dy_m=5.0,
        ),
    )


def apply_legacy_static_rho(path: Path, *, rho: float, static_rho: float) -> dict[str, float] | None:
    if static_rho <= 0.0 or np.isclose(static_rho, rho):
        return None

    import xarray as xr

    dataset = xr.open_dataset(path)
    try:
        patched = dataset.load()
    finally:
        dataset.close()

    scale = float(static_rho / rho)
    patched["hydrostatic_stiffness"] = patched["hydrostatic_stiffness"] * scale
    patched.attrs["hydrostatic_static_rho_kg_m3"] = float(static_rho)
    patched.attrs["hydrostatic_static_rho_scale"] = scale
    patched.attrs["hydrostatic_static_rho_note"] = (
        "hydrostatic_stiffness scaled after Capytaine generation for legacy Yoga compatibility"
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    patched.to_netcdf(tmp_path)
    patched.close()
    tmp_path.replace(path)
    return {"static_rho": float(static_rho), "scale": scale}


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    config: ArrayHydrodynamicsConfig,
    result,
    elapsed_seconds: float,
    logs: list[str],
    static_patch: dict[str, float] | None,
) -> Path:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(result.output_path),
        "sha256": sha256_file(result.output_path),
        "elapsed_seconds": elapsed_seconds,
        "body_count": result.body_count,
        "dof_count": result.dof_count,
        "problem_count": result.problem_count,
        "omega_count": result.omega_count,
        "wave_direction_count": result.wave_direction_count,
        "water_depth_m": result.water_depth_m,
        "omegas_rad_s": list(config.omegas_rad_s),
        "wave_directions_rad": list(config.wave_directions_rad),
        "module": {
            "length_m": config.module.length_m,
            "width_m": config.module.width_m,
            "height_m": config.module.height_m,
            "draft_m": config.module.draft_m,
            "mesh_size_m": config.module.mesh_size_m,
            "vertical_mesh_size_m": config.module.vertical_mesh_size_m,
            "mass_kg": config.module.mass_kg,
        },
        "rho_kg_m3": config.rho,
        "g_m_s2": config.g,
        "static_patch": static_patch,
        "compute_rao": config.compute_rao,
        "n_jobs": config.n_jobs,
        "layout_preview": preview_layout(config),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "logs": logs,
    }
    manifest_path = path.with_suffix(".generation_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    args.output = output
    if output.exists() and not args.force:
        print(f"Hydrodynamic dataset already exists: {output}")
        print("Use --force to regenerate it.")
        return 0

    config = build_config(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[hydro] DM10 Cummins hydrodynamic generation")
    print(f"[hydro] output: {output}")
    print(f"[hydro] omegas: {len(config.omegas_rad_s)} values, {config.omegas_rad_s}")
    print(f"[hydro] BEM problems: {len(config.omegas_rad_s) * (args.module_count * 6 + 1)}")
    logs: list[str] = []
    start = time.perf_counter()
    result = run_array_hydrodynamics(
        config,
        log=lambda message: (logs.append(message), print(f"[hydro] {message}", flush=True)),
    )
    elapsed_seconds = time.perf_counter() - start
    static_patch = apply_legacy_static_rho(output, rho=args.rho, static_rho=args.static_rho)
    manifest_path = write_manifest(
        output,
        args=args,
        config=config,
        result=result,
        elapsed_seconds=elapsed_seconds,
        logs=logs,
        static_patch=static_patch,
    )
    print("[hydro] completed")
    print(f"[hydro] elapsed_seconds: {elapsed_seconds:.3f}")
    print(f"[hydro] sha256: {sha256_file(output)}")
    print(f"[hydro] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
