"""Read-only NetCDF hydrodynamic dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from offshore_energy_sim.utils.hashing import sha256_file


@dataclass(frozen=True)
class HydrodynamicDatasetSummary:
    """Metadata and reproducibility information for a hydrodynamic NetCDF file."""

    path: Path
    exists: bool
    sha256: str | None
    xarray_available: bool
    capytaine_available: bool
    dims: dict[str, int] | None = None
    data_variables: tuple[str, ...] = ()
    coordinates: tuple[str, ...] = ()
    note: str | None = None


def xarray_available() -> bool:
    """Return whether xarray can be imported in the current environment."""

    return find_spec("xarray") is not None


def capytaine_xarray_available() -> bool:
    """Return whether Capytaine's xarray helpers can be imported."""

    try:
        return find_spec("capytaine.io.xarray") is not None
    except ModuleNotFoundError:
        return False


def open_hydrodynamic_dataset(
    path: str | Path,
    *,
    merge_complex: bool = True,
) -> Any:
    """Open a Capytaine-style hydrodynamic NetCDF dataset.

    This preserves the legacy loading convention used in `DM_Method.py`:
    `merge_complex_values(xr.open_dataset(path))`.
    """

    import xarray as xr

    dataset = xr.open_dataset(path)
    if merge_complex:
        from capytaine.io.xarray import merge_complex_values

        dataset = merge_complex_values(dataset)
    return dataset


def summarize_hydrodynamic_dataset(
    path: str | Path,
    *,
    load_metadata: bool = False,
) -> HydrodynamicDatasetSummary:
    """Summarize a hydrodynamic dataset without changing numerical data.

    If xarray/Capytaine are unavailable, the summary still reports existence
    and SHA-256 hash so baseline reproducibility can be checked.
    """

    path = Path(path)
    exists = path.exists()
    digest = sha256_file(path) if exists else None
    has_xarray = xarray_available()
    has_capytaine = capytaine_xarray_available()

    if not exists:
        return HydrodynamicDatasetSummary(
            path=path,
            exists=False,
            sha256=None,
            xarray_available=has_xarray,
            capytaine_available=has_capytaine,
            note="file does not exist",
        )

    if not load_metadata:
        return HydrodynamicDatasetSummary(
            path=path,
            exists=True,
            sha256=digest,
            xarray_available=has_xarray,
            capytaine_available=has_capytaine,
            note="metadata loading skipped",
        )

    if not has_xarray:
        return HydrodynamicDatasetSummary(
            path=path,
            exists=True,
            sha256=digest,
            xarray_available=False,
            capytaine_available=has_capytaine,
            note="xarray is not installed; only file hash was checked",
        )

    dataset = open_hydrodynamic_dataset(path, merge_complex=has_capytaine)
    try:
        dims = {name: int(size) for name, size in dataset.sizes.items()}
        return HydrodynamicDatasetSummary(
            path=path,
            exists=True,
            sha256=digest,
            xarray_available=has_xarray,
            capytaine_available=has_capytaine,
            dims=dims,
            data_variables=tuple(str(name) for name in dataset.data_vars),
            coordinates=tuple(str(name) for name in dataset.coords),
        )
    finally:
        dataset.close()
