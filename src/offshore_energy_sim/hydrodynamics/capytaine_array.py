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

    def resolved_mass_kg(self, rho: float) -> float:
        """Return explicit mass or neutral-buoyancy mass from displaced volume."""

        return self.mass_kg or self.length_m * self.width_m * self.draft_m * rho


@dataclass(frozen=True)
class ArrayLayoutSpec:
    """Rows, columns, and center-to-center spacing for a module array."""

    rows: int
    columns: int
    spacing_x_m: float
    spacing_y_m: float

    def __post_init__(self) -> None:
        _require_positive_int("rows", self.rows)
        _require_positive_int("columns", self.columns)
        _require_positive("spacing_x_m", self.spacing_x_m)
        _require_positive("spacing_y_m", self.spacing_y_m)

    @property
    def body_count(self) -> int:
        return self.rows * self.columns

    def module_centers(self) -> tuple[tuple[str, float, float], ...]:
        """Return ``(name, x, y)`` for every module in row-major order."""

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

    return {
        "body_count": config.layout.body_count,
        "dof_count": config.layout.body_count * 6,
        "omega_count": len(config.omegas_rad_s),
        "wave_direction_count": len(config.wave_directions_rad),
        "radiation_problem_count": config.layout.body_count * 6 * len(config.omegas_rad_s),
        "diffraction_problem_count": len(config.omegas_rad_s) * len(config.wave_directions_rad),
        "centers": [
            {"name": name, "x_m": x, "y_m": y}
            for name, x, y in config.layout.module_centers()
        ],
    }


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
    for name, x_m, y_m in config.layout.module_centers():
        mesh = mesh_parallelepiped(
            size=(config.module.length_m, config.module.width_m, config.module.height_m),
            resolution=config.module.mesh_resolution,
            center=(x_m, y_m, config.module.center_z_m),
            name=f"{name}_mesh",
        )
        body = cpt.FloatingBody(mesh=mesh, name=name)
        body.center_of_mass = (x_m, y_m, config.module.center_z_m)
        body.mass = config.module.resolved_mass_kg(config.rho)
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
        common["water_depth"] = config.water_depth_m

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
    results = solver.solve_all(problems, n_jobs=effective_n_jobs, progress_bar=False)

    attrs = {
        "rodm_hydrodynamics_generator": "offshore_energy_sim.hydrodynamics.capytaine_array",
        "module_length_m": config.module.length_m,
        "module_width_m": config.module.width_m,
        "module_height_m": config.module.height_m,
        "module_draft_m": config.module.draft_m,
        "module_mass_kg": config.module.resolved_mass_kg(config.rho),
        "array_rows": config.layout.rows,
        "array_columns": config.layout.columns,
        "array_spacing_x_m": config.layout.spacing_x_m,
        "array_spacing_y_m": config.layout.spacing_y_m,
    }
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
            added_mass_by_dof = result.added_mass
            damping_by_dof = result.radiation_damping
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
