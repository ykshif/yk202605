"""Check whether the local Python environment can run the refactored platform."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = {
    "numpy": "Numerical arrays",
    "scipy": "Linear algebra and sparse solvers",
    "xarray": "Hydrodynamic NetCDF datasets",
    "yaml": "YAML case configuration",
    "matplotlib": "Validation figures",
    "capytaine": "Hydrodynamic data compatibility",
}

OPTIONAL_MODULES = {
    "netCDF4": "NetCDF backend",
    "h5netcdf": "Alternative NetCDF backend",
    "h5py": "HDF5 support",
    "pandas": "Tabular utilities and pvlib dependency",
    "vtk": "Legacy visualization scripts",
    "pvlib": "PV power workflows",
    "jupyterlab": "Notebook work",
}


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def _print_group(title: str, modules: dict[str, str]) -> list[str]:
    missing = []
    print(title)
    for name, purpose in modules.items():
        try:
            version = _module_version(name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            missing.append(name)
            print(f"  missing: {name} ({purpose}) [{exc.__class__.__name__}: {exc}]")
        else:
            print(f"  ok: {name} {version} ({purpose})")
    return missing


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    required_missing = _print_group("Required modules:", REQUIRED_MODULES)
    optional_missing = _print_group("Optional modules:", OPTIONAL_MODULES)

    if optional_missing:
        print("")
        print("Optional modules missing:")
        for name in optional_missing:
            print(f"  - {name}")

    if required_missing:
        print("")
        print("Environment check failed. Missing required modules:")
        for name in required_missing:
            print(f"  - {name}")
        return 1

    print("")
    print("Environment check passed for core RODM workflows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
