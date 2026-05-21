"""Generate Capytaine hydrodynamic datasets for rectangular module arrays."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Iterable, Sequence
import math
import re

import numpy as np


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class RectangularModuleSpec:
    """Basic geometry and mesh settings for one rectangular floating module."""

    length_m: float
    width_m: float
    height_m: float
    draft_m: float
    mesh_size_m: float
    vertical_mesh_size_m: float | None = None
    mass_kg: float | None = None

    def __post_init__(self) -> None:
        _require_positive("length_m", self.length_m)
        _require_positive("width_m", self.width_m)
        _require_positive("height_m", self.height_m)
        _require_positive("draft_m", self.draft_m)
        _require_positive("mesh_size_m", self.mesh_size_m)
        if self.vertical_mesh_size_m is not None:
            _require_positive("vertical_mesh_size_m", self.vertical_mesh_size_m)
        if self.mass_kg is not None:
            _require_positive("mass_kg", self.mass_kg)
        if self.draft_m > self.height_m:
            raise ValueError("draft_m must be smaller than or equal to height_m")

    @property
    def center_z_m(self) -> float:
        """Body center elevation for the requested draft and free surface z=0."""

        return 0.5 * self.height_m - self.draft_m

    @property
    def mesh_resolution(self) -> tuple[int, int, int]:
        """Panel resolution tuple used by Capytaine's parallelepiped mesher."""

        vertical_size = self.vertical_mesh_size_m or self.mesh_size_m
        return (
            max(1, int(self.length_m / self.mesh_size_m)),
            max(1, int(self.width_m / self.mesh_size_m)),
            max(1, int(self.height_m / vertical_size)),
        )

    def mesh_resolution_for_length(self, length_m: float) -> tuple[int, int, int]:
        """Panel resolution tuple for a module with the requested length."""

        _require_positive("length_m", length_m)
        vertical_size = self.vertical_mesh_size_m or self.mesh_size_m
        return (
            max(1, int(length_m / self.mesh_size_m)),
            max(1, int(self.width_m / self.mesh_size_m)),
            max(1, int(self.height_m / vertical_size)),
        )

    def resolved_mass_kg_for_length(self, rho: float, length_m: float) -> float:
        """Return explicit mass or neutral-buoyancy mass for one module length."""

        _require_positive("length_m", length_m)
        return self.mass_kg or length_m * self.width_m * self.draft_m * rho

    def resolved_mass_kg(self, rho: float) -> float:
        """Return explicit mass or neutral-buoyancy mass from displaced volume."""

        return self.resolved_mass_kg_for_length(rho, self.length_m)


@dataclass(frozen=True)
class ModuleGeometrySpec:
    """One rectangular module in the array layout."""

    name: str
    x_start_m: float
    x_end_m: float
    x_m: float
    y_m: float
    length_m: float


@dataclass(frozen=True)
class StructuralGridSpec:
    """Regular FEM grid used to map hydrodynamic module centers to nodes.

    The 300 m x 60 m reference structure uses a 5 m x 5 m grid. With
    row-major Abaqus numbering along x, this gives 61 x 13 = 793 nodes. The
    y-origin is the port/starboard edge, so the centerline y=0 maps to row 6.
    """

    length_m: float = 300.0
    width_m: float = 60.0
    dx_m: float = 5.0
    dy_m: float = 5.0
    x_origin_m: float = 0.0
    y_origin_m: float | None = None
    x_axis_reversed_in_node_numbering: bool = True
    tolerance_m: float = 1.0e-6

    def __post_init__(self) -> None:
        _require_positive("structural_grid_length_m", self.length_m)
        _require_positive("structural_grid_width_m", self.width_m)
        _require_positive("structural_grid_dx_m", self.dx_m)
        _require_positive("structural_grid_dy_m", self.dy_m)
        _require_positive("structural_grid_tolerance_m", self.tolerance_m)

    @property
    def y0_m(self) -> float:
        return -0.5 * self.width_m if self.y_origin_m is None else float(self.y_origin_m)

    @property
    def nodes_per_x(self) -> int:
        return _grid_count(self.length_m, self.dx_m, "length_m/dx_m")

    @property
    def nodes_per_y(self) -> int:
        return _grid_count(self.width_m, self.dy_m, "width_m/dy_m")

    @property
    def total_nodes(self) -> int:
        return self.nodes_per_x * self.nodes_per_y

    def node_for_point(self, x_m: float, y_m: float) -> dict[str, float | int]:
        """Return one-based FEM node metadata for a grid-aligned point."""

        x_index = _grid_index(x_m - self.x_origin_m, self.dx_m, "module center x", self.tolerance_m)
        y_index = _grid_index(y_m - self.y0_m, self.dy_m, "module center y", self.tolerance_m)
        if not 0 <= x_index < self.nodes_per_x:
            raise ValueError("module center x is outside the structural grid")
        if not 0 <= y_index < self.nodes_per_y:
            raise ValueError("module center y is outside the structural grid")
        node_x_index = self.nodes_per_x - 1 - x_index if self.x_axis_reversed_in_node_numbering else x_index
        return {
            "fem_node_one_based": y_index * self.nodes_per_x + node_x_index + 1,
            "x_index": x_index,
            "node_x_index": node_x_index,
            "y_index": y_index,
            "x_m": self.x_origin_m + x_index * self.dx_m,
            "y_m": self.y0_m + y_index * self.dy_m,
        }


@dataclass(frozen=True)
class ArrayLayoutSpec:
    """Rows, columns, and center-to-center spacing for a module array."""

    rows: int
    columns: int
    spacing_x_m: float
    spacing_y_m: float
    division_mode: str = "uniform"
    total_length_m: float | None = None
    module_lengths_x_m: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_positive_int("rows", self.rows)
        _require_positive_int("columns", self.columns)
        _require_positive("spacing_x_m", self.spacing_x_m)
        _require_positive("spacing_y_m", self.spacing_y_m)
        mode = self.division_mode.lower().replace("-", "_")
        if mode not in {"uniform", "custom", "random"}:
            raise ValueError("division_mode must be uniform, custom, or random")
        object.__setattr__(self, "division_mode", mode)
        if self.total_length_m is not None:
            _require_positive("total_length_m", self.total_length_m)
        if self.module_lengths_x_m is not None:
            lengths = tuple(float(value) for value in self.module_lengths_x_m)
            if len(lengths) != self.columns:
                raise ValueError("module_lengths_x_m length must equal columns")
            for value in lengths:
                _require_positive("module_length_m", value)
            if mode in {"custom", "random"} and self.rows != 1:
                raise ValueError("custom/random division is one-dimensional and requires rows=1")
            if self.total_length_m is not None and not np.isclose(
                sum(lengths),
                self.total_length_m,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError("sum(module_lengths_x_m) must equal total_length_m")
            object.__setattr__(self, "module_lengths_x_m", lengths)

    @property
    def body_count(self) -> int:
        return self.rows * self.columns

    def module_lengths(self, default_length_m: float) -> tuple[float, ...]:
        """Return module lengths along x for one row."""

        _require_positive("default_length_m", default_length_m)
        if self.module_lengths_x_m is not None:
            return self.module_lengths_x_m
        return tuple(float(default_length_m) for _ in range(self.columns))

    def x_boundaries(self, default_length_m: float) -> tuple[float, ...]:
        """Return cumulative x boundaries from 0 to the total length."""

        boundaries = [0.0]
        for length in self.module_lengths(default_length_m):
            boundaries.append(boundaries[-1] + length)
        return tuple(boundaries)

    def module_geometries(self, default_length_m: float) -> tuple[ModuleGeometrySpec, ...]:
        """Return geometry for every module in row-major order."""

        if self.module_lengths_x_m is None:
            return tuple(
                ModuleGeometrySpec(
                    name=name,
                    x_start_m=x - 0.5 * default_length_m,
                    x_end_m=x + 0.5 * default_length_m,
                    x_m=x,
                    y_m=y,
                    length_m=default_length_m,
                )
                for name, x, y in self.module_centers()
            )

        boundaries = self.x_boundaries(default_length_m)
        y0 = 0.5 * (self.rows - 1) * self.spacing_y_m
        geometries: list[ModuleGeometrySpec] = []
        for row in range(self.rows):
            for column in range(self.columns):
                x_start = boundaries[column]
                x_end = boundaries[column + 1]
                name = f"{column}_{row}"
                y = y0 - row * self.spacing_y_m
                geometries.append(
                    ModuleGeometrySpec(
                        name=name,
                        x_start_m=x_start,
                        x_end_m=x_end,
                        x_m=0.5 * (x_start + x_end),
                        y_m=y,
                        length_m=x_end - x_start,
                    )
                )
        return tuple(geometries)

    def module_centers(self) -> tuple[tuple[str, float, float], ...]:
        """Return ``(name, x, y)`` for every module in row-major order."""

        if self.module_lengths_x_m is not None:
            return tuple(
                (item.name, item.x_m, item.y_m)
                for item in self.module_geometries(self.module_lengths_x_m[0])
            )

        centers: list[tuple[str, float, float]] = []
        x0 = -0.5 * (self.columns - 1) * self.spacing_x_m
        y0 = 0.5 * (self.rows - 1) * self.spacing_y_m
        for row in range(self.rows):
            for column in range(self.columns):
                name = f"{column}_{row}"
                x = x0 + column * self.spacing_x_m
                y = y0 - row * self.spacing_y_m
                centers.append((name, x, y))
        return tuple(centers)


@dataclass(frozen=True)
class ArrayHydrodynamicsConfig:
    """Complete input deck for a rectangular-array Capytaine run."""

    module: RectangularModuleSpec
    layout: ArrayLayoutSpec
    omegas_rad_s: tuple[float, ...]
    output_path: Path
    wave_directions_rad: tuple[float, ...] = (0.0,)
    water_depth_m: float | None = None
    rho: float = 1025.0
    g: float = 9.81
    n_jobs: int = 1
    compute_rao: bool = True
    structural_grid: StructuralGridSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))
        if len(self.omegas_rad_s) == 0:
            raise ValueError("omegas_rad_s must contain at least one frequency")
        if len(self.wave_directions_rad) == 0:
            raise ValueError("wave_directions_rad must contain at least one direction")
        for omega in self.omegas_rad_s:
            _require_positive("omega", omega)
        for direction in self.wave_directions_rad:
            _require_finite("wave_direction_rad", direction)
        if self.water_depth_m is not None:
            _require_positive("water_depth_m", self.water_depth_m)
            if self.water_depth_m <= self.module.draft_m:
                raise ValueError("water_depth_m must be larger than module draft_m")
        _require_positive("rho", self.rho)
        _require_positive("g", self.g)
        _require_positive_int("n_jobs", self.n_jobs)
        if self.structural_grid is not None:
            module_structural_node_mappings(self)


@dataclass(frozen=True)
class ArrayHydrodynamicsResult:
    """Small summary of a completed hydrodynamic dataset generation."""

    output_path: Path
    body_count: int
    dof_count: int
    problem_count: int
    omega_count: int
    wave_direction_count: int
    water_depth_m: float | None
    rao_preview: dict[str, object] | None = None


def omega_values_from_range(start: float, stop: float, count: int) -> tuple[float, ...]:
    """Build an inclusive angular-frequency grid in rad/s."""

    _require_positive("start", start)
    _require_positive("stop", stop)
    _require_positive_int("count", count)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    return tuple(float(value) for value in np.linspace(start, stop, count))


def omega_from_wavelength(wavelength_m: float, water_depth_m: float | None, g: float = 9.81) -> float:
    """Return angular frequency from wavelength using the linear dispersion relation."""

    _require_positive("wavelength_m", wavelength_m)
    _require_positive("g", g)
    if water_depth_m is not None:
        _require_positive("water_depth_m", water_depth_m)
    wavenumber = 2.0 * np.pi / wavelength_m
    depth_factor = 1.0 if water_depth_m is None else np.tanh(wavenumber * water_depth_m)
    return float(np.sqrt(g * wavenumber * depth_factor))


def omega_values_from_wavelengths(
    wavelengths_m: str | Iterable[float],
    water_depth_m: float | None,
    g: float = 9.81,
) -> tuple[float, ...]:
    """Convert one or more wavelengths in meters to angular frequencies."""

    wavelengths = parse_float_sequence(wavelengths_m)
    return tuple(omega_from_wavelength(value, water_depth_m, g) for value in wavelengths)


def parse_float_sequence(values: str | Iterable[float]) -> tuple[float, ...]:
    """Parse comma, semicolon, whitespace, or newline separated floats."""

    if isinstance(values, str):
        tokens = [token for token in re.split(r"[\s,;]+", values.strip()) if token]
        return tuple(float(token) for token in tokens)
    return tuple(float(value) for value in values)


def degrees_to_radians(values: Sequence[float]) -> tuple[float, ...]:
    """Convert wave directions from degrees to radians."""

    return tuple(float(np.deg2rad(value)) for value in values)


def preview_layout(config: ArrayHydrodynamicsConfig) -> dict[str, object]:
    """Return UI-friendly array metadata without starting a BEM solve."""

    preview = {
        "body_count": config.layout.body_count,
        "dof_count": config.layout.body_count * 6,
        "omega_count": len(config.omegas_rad_s),
        "wave_direction_count": len(config.wave_directions_rad),
        "radiation_problem_count": config.layout.body_count * 6 * len(config.omegas_rad_s),
        "diffraction_problem_count": len(config.omegas_rad_s) * len(config.wave_directions_rad),
        "division_mode": config.layout.division_mode,
        "module_lengths_x_m": list(config.layout.module_lengths(config.module.length_m)),
        "x_boundaries_m": list(config.layout.x_boundaries(config.module.length_m)),
        "centers": [
            {
                "name": item.name,
                "x_start_m": item.x_start_m,
                "x_end_m": item.x_end_m,
                "x_m": item.x_m,
                "y_m": item.y_m,
                "length_m": item.length_m,
            }
            for item in config.layout.module_geometries(config.module.length_m)
        ],
    }
    if config.structural_grid is not None:
        preview["structural_grid"] = {
            "dx_m": config.structural_grid.dx_m,
            "dy_m": config.structural_grid.dy_m,
            "nodes_per_x": config.structural_grid.nodes_per_x,
            "nodes_per_y": config.structural_grid.nodes_per_y,
            "total_nodes": config.structural_grid.total_nodes,
        }
        preview["structural_nodes"] = module_structural_node_mappings(config)
    return preview


def module_structural_node_mappings(config: ArrayHydrodynamicsConfig) -> list[dict[str, object]]:
    """Map each hydrodynamic module center to a one-based FEM node ID.

    The mapping is metadata only: hydrodynamic matrices keep the same module
    order, and the RODM solver can use these node IDs to build the matching
    structural master-node order.
    """

    if config.structural_grid is None:
        return []
    x_offset = _structural_x_offset(config)
    mappings: list[dict[str, object]] = []
    for geometry in config.layout.module_geometries(config.module.length_m):
        node = config.structural_grid.node_for_point(
            geometry.x_m + x_offset,
            geometry.y_m,
        )
        mappings.append(
            {
                "name": geometry.name,
                "fem_node_one_based": node["fem_node_one_based"],
                "x_index": node["x_index"],
                "node_x_index": node["node_x_index"],
                "y_index": node["y_index"],
                "x_m": node["x_m"],
                "y_m": node["y_m"],
                "hydrodynamic_x_m": geometry.x_m,
                "hydrodynamic_y_m": geometry.y_m,
            }
        )
    return mappings


def _structural_x_offset(config: ArrayHydrodynamicsConfig) -> float:
    """Offset origin-centered legacy uniform modules onto the 0..L FEM grid."""

    if config.layout.module_lengths_x_m is not None:
        return 0.0
    total_length = (
        config.layout.total_length_m
        if config.layout.total_length_m is not None
        else config.layout.columns * config.module.length_m
    )
    return 0.5 * total_length


def summarize_rao_for_ui(
    rao_data,
    *,
    omega_index: int = 0,
    wave_direction_index: int = 0,
) -> dict[str, object]:
    """Convert Capytaine RAO data into compact JSON-ready motion amplitudes."""

    selected = rao_data
    if "omega" in selected.dims:
        selected = selected.isel(omega=omega_index)
    if "wave_direction" in selected.dims:
        selected = selected.isel(wave_direction=wave_direction_index)

    body_map: dict[str, dict[str, object]] = {}
    for dof_label in selected.coords["radiating_dof"].values:
        label = str(dof_label)
        if "__" in label:
            body_name, dof_name = label.split("__", 1)
        else:
            body_name, dof_name = "body", label
        value = complex(selected.sel(radiating_dof=dof_label).values)
        body_entry = body_map.setdefault(body_name, {"name": body_name, "dofs": {}})
        body_entry["dofs"][dof_name] = _complex_summary(value)

    preview = {
        "omega_rad_s": _coord_float(rao_data, "omega", omega_index),
        "wave_direction_rad": _coord_float(rao_data, "wave_direction", wave_direction_index),
        "body_count": len(body_map),
        "bodies": list(body_map.values()),
    }
    return preview


def build_array_body(config: ArrayHydrodynamicsConfig):
    """Create a Capytaine FloatingBody with RODM-compatible DOF labels."""

    cpt, mesh_parallelepiped = _require_capytaine()
    bodies = []
    for geometry in config.layout.module_geometries(config.module.length_m):
        mesh = mesh_parallelepiped(
            size=(geometry.length_m, config.module.width_m, config.module.height_m),
            resolution=config.module.mesh_resolution_for_length(geometry.length_m),
            center=(geometry.x_m, geometry.y_m, config.module.center_z_m),
            name=f"{geometry.name}_mesh",
        )
        body = cpt.FloatingBody(mesh=mesh, name=geometry.name)
        body.center_of_mass = (geometry.x_m, geometry.y_m, config.module.center_z_m)
        body.mass = config.module.resolved_mass_kg_for_length(config.rho, geometry.length_m)
        body.add_all_rigid_body_dofs()
        body.inertia_matrix = body.compute_rigid_body_inertia(rho=config.rho)
        body.keep_immersed_part(free_surface=0.0)
        body.rotation_center = body.center_of_mass
        body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(
            rho=config.rho,
            g=config.g,
        )
        bodies.append(body)

    return cpt.FloatingBody.join_bodies(
        *bodies,
        name=f"array_{config.layout.columns}x{config.layout.rows}",
    )


def build_hydrodynamic_problems(config: ArrayHydrodynamicsConfig, body) -> list[object]:
    """Create radiation and diffraction problems for the full input deck."""

    cpt, _ = _require_capytaine()
    common = {
        "body": body,
        "rho": config.rho,
        "g": config.g,
    }
    if config.water_depth_m is not None:
        from inspect import signature

        radiation_parameters = signature(cpt.RadiationProblem).parameters
        if "water_depth" in radiation_parameters:
            common["water_depth"] = config.water_depth_m
        else:
            common["sea_bottom"] = -config.water_depth_m

    problems: list[object] = []
    for omega in config.omegas_rad_s:
        for dof in body.dofs:
            problems.append(
                cpt.RadiationProblem(
                    omega=omega,
                    radiating_dof=dof,
                    **common,
                )
            )
        for wave_direction in config.wave_directions_rad:
            problems.append(
                cpt.DiffractionProblem(
                    omega=omega,
                    wave_direction=wave_direction,
                    **common,
                )
            )
    return problems


def run_array_hydrodynamics(
    config: ArrayHydrodynamicsConfig,
    *,
    log: LogCallback | None = None,
) -> ArrayHydrodynamicsResult:
    """Run Capytaine and write a separated-complex-value NetCDF dataset."""

    cpt, _ = _require_capytaine()
    _log(log, "Building rectangular module array geometry")
    body = build_array_body(config)
    problems = build_hydrodynamic_problems(config, body)
    _log(
        log,
        (
            "Solving "
            f"{len(problems)} BEM problems "
            f"({len(config.omegas_rad_s)} omega, "
            f"{len(config.wave_directions_rad)} wave direction, "
            f"{len(body.dofs)} DOF)"
        ),
    )

    effective_n_jobs = _effective_n_jobs(config.n_jobs, log=log)
    solver = cpt.BEMSolver()
    from inspect import signature

    if "progress_bar" in signature(solver.solve_all).parameters:
        results = solver.solve_all(problems, n_jobs=effective_n_jobs, progress_bar=False)
    else:
        results = solver.solve_all(problems, n_jobs=effective_n_jobs)

    attrs = {
        "rodm_hydrodynamics_generator": "offshore_energy_sim.hydrodynamics.capytaine_array",
        "rho_kg_m3": config.rho,
        "gravity_m_s2": config.g,
        "module_length_m": config.module.length_m,
        "module_width_m": config.module.width_m,
        "module_height_m": config.module.height_m,
        "module_draft_m": config.module.draft_m,
        "module_mesh_size_m": config.module.mesh_size_m,
        "module_vertical_mesh_size_m": config.module.vertical_mesh_size_m
        or config.module.mesh_size_m,
        "module_mass_kg": config.module.resolved_mass_kg(config.rho),
        "array_rows": config.layout.rows,
        "array_columns": config.layout.columns,
        "array_spacing_x_m": config.layout.spacing_x_m,
        "array_spacing_y_m": config.layout.spacing_y_m,
        "array_division_mode": config.layout.division_mode,
        "array_total_length_m": config.layout.x_boundaries(config.module.length_m)[-1],
        "array_module_lengths_x_m": ",".join(
            str(value) for value in config.layout.module_lengths(config.module.length_m)
        ),
    }
    if config.water_depth_m is not None:
        attrs["water_depth_m"] = config.water_depth_m
    if config.structural_grid is not None:
        mappings = module_structural_node_mappings(config)
        attrs.update(
            {
                "structural_grid_dx_m": config.structural_grid.dx_m,
                "structural_grid_dy_m": config.structural_grid.dy_m,
                "structural_grid_nodes_per_x": config.structural_grid.nodes_per_x,
                "structural_grid_nodes_per_y": config.structural_grid.nodes_per_y,
                "structural_grid_total_nodes": config.structural_grid.total_nodes,
                "array_structural_node_ids": ",".join(
                    str(item["fem_node_one_based"]) for item in mappings
                ),
                "array_structural_node_x_indices": ",".join(
                    str(item["x_index"]) for item in mappings
                ),
                "array_structural_node_y_indices": ",".join(
                    str(item["y_index"]) for item in mappings
                ),
                "array_structural_node_x_m": ",".join(str(item["x_m"]) for item in mappings),
                "array_structural_node_y_m": ",".join(str(item["y_m"]) for item in mappings),
            }
        )
    _log(log, "Assembling Capytaine xarray dataset")
    if config.compute_rao:
        dataset = cpt.assemble_dataset(
            results,
            freq=False,
            wavenumber=False,
            wavelength=False,
            period=False,
            mesh=False,
            attrs=attrs,
        )
    else:
        dataset = _assemble_rodm_dataset(results, body=body, config=config, attrs=attrs)
    rao_preview = _add_rao_to_dataset(dataset, log=log) if config.compute_rao else None
    if not config.compute_rao:
        _log(log, "RAO computation skipped by configuration")

    output_path = config.output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(log, f"Writing NetCDF dataset: {output_path}")
    _prepare_dataset_for_netcdf(cpt.io.xarray.separate_complex_values(dataset)).to_netcdf(
        output_path
    )
    _log(log, "Hydrodynamic NetCDF generation completed")

    return ArrayHydrodynamicsResult(
        output_path=output_path,
        body_count=config.layout.body_count,
        dof_count=len(body.dofs),
        problem_count=len(problems),
        omega_count=len(config.omegas_rad_s),
        wave_direction_count=len(config.wave_directions_rad),
        water_depth_m=config.water_depth_m,
        rao_preview=rao_preview,
    )


def _assemble_rodm_dataset(results, *, body, config: ArrayHydrodynamicsConfig, attrs: dict):
    """Assemble only the hydrodynamic terms needed by the RODM solver.

    Capytaine's generic dataframe-based assembler is convenient, but for large
    single-frequency arrays it can dominate the runtime.  This direct assembler
    keeps the same core variable names and dimensions used by the solver while
    avoiding optional coordinates that are not consumed downstream.
    """

    import xarray as xr

    dof_labels = tuple(str(label) for label in body.dofs.keys())
    dof_index = {label: index for index, label in enumerate(dof_labels)}
    omega_values = np.asarray(config.omegas_rad_s, dtype=float)
    wave_direction_values = np.asarray(config.wave_directions_rad, dtype=float)
    n_omega = len(omega_values)
    n_wave = len(wave_direction_values)
    n_dof = len(dof_labels)

    added_mass = np.full((n_omega, n_dof, n_dof), np.nan, dtype=float)
    radiation_damping = np.full((n_omega, n_dof, n_dof), np.nan, dtype=float)
    diffraction_force = np.full((n_omega, n_wave, n_dof), np.nan + 0.0j, dtype=complex)
    froude_krylov_force = np.full((n_omega, n_wave, n_dof), np.nan + 0.0j, dtype=complex)

    for result in results:
        omega_index = _nearest_index(float(result.omega), omega_values, "omega")
        if hasattr(result, "radiating_dof"):
            radiating_index = dof_index[str(result.radiating_dof)]
            if hasattr(result, "added_mass"):
                added_mass_by_dof = result.added_mass
                damping_by_dof = result.radiation_damping
            else:
                added_mass_by_dof = result.added_masses
                damping_by_dof = result.radiation_dampings
            for label in dof_labels:
                influenced_index = dof_index[label]
                added_mass[omega_index, radiating_index, influenced_index] = float(
                    added_mass_by_dof[label]
                )
                radiation_damping[omega_index, radiating_index, influenced_index] = float(
                    damping_by_dof[label]
                )
        else:
            from capytaine.bem.problems_and_results import froude_krylov_force as fk_force

            wave_index = _nearest_index(
                float(result.wave_direction),
                wave_direction_values,
                "wave_direction",
            )
            fk_by_dof = fk_force(result.problem)
            for label in dof_labels:
                influenced_index = dof_index[label]
                diffraction_force[omega_index, wave_index, influenced_index] = complex(
                    result.forces[label]
                )
                froude_krylov_force[omega_index, wave_index, influenced_index] = complex(
                    fk_by_dof[label]
                )

    _require_filled("added_mass", added_mass)
    _require_filled("radiation_damping", radiation_damping)
    _require_filled("diffraction_force", diffraction_force)
    _require_filled("Froude_Krylov_force", froude_krylov_force)

    coords = {
        "omega": omega_values,
        "wave_direction": wave_direction_values,
        "radiating_dof": list(dof_labels),
        "influenced_dof": list(dof_labels),
    }
    dataset = xr.Dataset(
        data_vars={
            "added_mass": (
                ("omega", "radiating_dof", "influenced_dof"),
                added_mass,
            ),
            "radiation_damping": (
                ("omega", "radiating_dof", "influenced_dof"),
                radiation_damping,
            ),
            "diffraction_force": (
                ("omega", "wave_direction", "influenced_dof"),
                diffraction_force,
            ),
            "Froude_Krylov_force": (
                ("omega", "wave_direction", "influenced_dof"),
                froude_krylov_force,
            ),
            "excitation_force": (
                ("omega", "wave_direction", "influenced_dof"),
                froude_krylov_force + diffraction_force,
            ),
            "inertia_matrix": (
                ("influenced_dof", "radiating_dof"),
                _body_matrix_values(body.inertia_matrix, dof_labels),
            ),
            "hydrostatic_stiffness": (
                ("influenced_dof", "radiating_dof"),
                _body_matrix_values(body.hydrostatic_stiffness, dof_labels),
            ),
        },
        coords=coords,
        attrs=dict(attrs),
    )
    dataset.attrs["capytaine_version"] = getattr(_require_capytaine()[0], "__version__", "")
    return dataset


def _require_capytaine():
    try:
        import capytaine as cpt
        from capytaine.meshes.predefined import mesh_parallelepiped
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Capytaine is required to generate hydrodynamic NetCDF files. "
            "Use the offshore-energy-sim conda environment."
        ) from exc
    return cpt, mesh_parallelepiped


def _nearest_index(value: float, values: np.ndarray, name: str) -> int:
    index = int(np.argmin(np.abs(values - value)))
    if not np.isclose(values[index], value):
        raise ValueError(f"Unexpected {name} value {value!r}; expected one of {values!r}")
    return index


def _require_filled(name: str, values: np.ndarray) -> None:
    if np.isnan(np.asarray(values).real).any() or np.isnan(np.asarray(values).imag).any():
        raise ValueError(f"Incomplete hydrodynamic assembly for {name}")


def _body_matrix_values(matrix, dof_labels: Sequence[str]) -> np.ndarray:
    if hasattr(matrix, "sel") and hasattr(matrix, "dims"):
        if "influenced_dof" in matrix.dims and "radiating_dof" in matrix.dims:
            matrix = matrix.sel(influenced_dof=list(dof_labels), radiating_dof=list(dof_labels))
    return np.asarray(getattr(matrix, "values", matrix), dtype=float)


def _prepare_dataset_for_netcdf(dataset):
    """Convert Capytaine categorical DOF coordinates to NetCDF-safe strings."""

    for coord_name in ("radiating_dof", "influenced_dof"):
        if coord_name in dataset.coords:
            dataset = dataset.assign_coords(
                {coord_name: [str(value) for value in dataset[coord_name].values]}
            )
    return dataset


def _effective_n_jobs(n_jobs: int, *, log: LogCallback | None = None) -> int:
    """Use serial solving when Capytaine's optional parallel dependency is missing."""

    if n_jobs > 1 and find_spec("joblib") is None:
        _log(
            log,
            (
                "joblib is not installed; falling back to n_jobs=1. "
                "Install joblib to enable Capytaine parallel solve_all."
            ),
        )
        return 1
    return n_jobs


def _add_rao_to_dataset(dataset, *, log: LogCallback | None = None) -> dict[str, object] | None:
    try:
        from capytaine.post_pro.rao import rao

        rao_data = rao(dataset)
    except Exception as exc:  # pragma: no cover - depends on BEM matrix conditioning
        _log(log, f"RAO computation skipped: {exc}")
        return None

    dataset["rao"] = rao_data
    _log(log, "Computed RAO and added it to the NetCDF dataset")
    return summarize_rao_for_ui(rao_data)


def _complex_summary(value: complex) -> dict[str, float]:
    return {
        "re": float(np.real(value)),
        "im": float(np.imag(value)),
        "abs": float(np.abs(value)),
        "phase_rad": float(np.angle(value)),
    }


def _coord_float(data_array, coord_name: str, index: int) -> float | None:
    if coord_name not in data_array.coords:
        return None
    coord = data_array.coords[coord_name]
    if coord.ndim == 0:
        return float(coord.values)
    return float(coord.values[index])


def _log(log: LogCallback | None, message: str) -> None:
    if log is not None:
        log(message)


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_int(name: str, value: int) -> None:
    if int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _grid_count(length: float, spacing: float, name: str) -> int:
    intervals = length / spacing
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be an integer")
    return int(rounded) + 1


def _grid_index(distance: float, spacing: float, name: str, tolerance: float) -> int:
    index = distance / spacing
    rounded = round(index)
    if not math.isclose(index, rounded, rel_tol=0.0, abs_tol=tolerance / spacing):
        raise ValueError(f"{name} must lie on a structural grid node")
    return int(rounded)
