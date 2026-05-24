"""Dataset IO and plotting for hydrodynamic extrapolation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset
from offshore_energy_sim.time_domain_adapter.hydrodynamic_extrapolation import (
    HydrodynamicExtrapolationConfig,
    extrapolate_frequency_series,
    max_abs_difference_inside_original_range,
)


FORCE_VARIABLES = {
    "Froude_Krylov_force",
    "diffraction_force",
    "excitation_force",
}


def _omega_first(array, omega_dim: str = "omega"):
    if omega_dim not in array.dims:
        raise ValueError(f"{array.name} does not contain an omega dimension")
    other_dims = [dim for dim in array.dims if dim != omega_dim]
    transposed = array.transpose(omega_dim, *other_dims)
    return transposed, other_dims


def extend_hydrodynamic_xarray_dataset(dataset, config: HydrodynamicExtrapolationConfig):
    """Return a new xarray dataset with extrapolated omega-dependent variables."""

    import xarray as xr

    omega = np.asarray(dataset["omega"].values, dtype=float).reshape(-1)
    extended_omega, _, original_slice = extrapolate_frequency_series(
        omega,
        dataset["added_mass"].values,
        config=config,
        series_kind="added_mass",
    )
    coords = {name: coord.copy(deep=True) for name, coord in dataset.coords.items() if name != "omega"}
    coords["omega"] = extended_omega
    data_vars = {}
    invariance: dict[str, float] = {}

    for name, variable in dataset.data_vars.items():
        if "omega" not in variable.dims:
            data_vars[name] = variable.copy(deep=True)
            continue

        omega_first, other_dims = _omega_first(variable)
        values = np.asarray(omega_first.values)
        if name == "added_mass":
            _, extrapolated, variable_slice = extrapolate_frequency_series(
                omega,
                values,
                config=config,
                series_kind="added_mass",
            )
        elif name == "radiation_damping":
            _, extrapolated, variable_slice = extrapolate_frequency_series(
                omega,
                values,
                config=config,
                series_kind="radiation_damping",
            )
        elif name in FORCE_VARIABLES:
            _, extrapolated, variable_slice = extrapolate_frequency_series(
                omega,
                values,
                config=config,
                series_kind="force",
            )
        else:
            _, extrapolated, variable_slice = extrapolate_frequency_series(
                omega,
                values,
                config=config,
                series_kind="force",
            )
        invariance[name] = max_abs_difference_inside_original_range(values, extrapolated, variable_slice)

        dims = ("omega", *other_dims)
        data_array = xr.DataArray(extrapolated, dims=dims)
        data_array = data_array.transpose(*variable.dims)
        data_vars[name] = data_array

    extended = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(dataset.attrs))
    extended.attrs["time_domain_adapter_extrapolated"] = "true"
    extended.attrs["time_domain_adapter_note"] = (
        "Omega-dependent hydrodynamic variables were extrapolated outside the "
        "original frequency band. Embedded original frequencies are unchanged."
    )
    extended.attrs["time_domain_adapter_original_omega_min"] = float(omega[0])
    extended.attrs["time_domain_adapter_original_omega_max"] = float(omega[-1])
    extended.attrs["time_domain_adapter_extended_omega_min"] = float(extended_omega[0])
    extended.attrs["time_domain_adapter_extended_omega_max"] = float(extended_omega[-1])
    extended.attrs["time_domain_adapter_config"] = repr(config.to_dict())
    return extended, invariance, original_slice


def write_extrapolated_hydrodynamic_dataset(
    source_path: str | Path,
    output_path: str | Path,
    config: HydrodynamicExtrapolationConfig,
) -> tuple[Path, dict[str, float]]:
    """Write an extrapolated copy of a Capytaine-style hydrodynamic NetCDF file."""

    import xarray as xr

    source = Path(source_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.open_dataset(source)
    try:
        extended, invariance, _ = extend_hydrodynamic_xarray_dataset(dataset, config)
        try:
            extended.to_netcdf(target)
        finally:
            extended.close()
    finally:
        dataset.close()
    return target, invariance


def load_merged_hydrodynamic_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load merged complex hydrodynamic arrays for diagnostics."""

    dataset = open_hydrodynamic_dataset(path, merge_complex=True)
    try:
        froude = dataset["Froude_Krylov_force"]
        diffraction = dataset["diffraction_force"]
        if "wave_direction" in froude.dims:
            froude = froude.isel(wave_direction=0)
            diffraction = diffraction.isel(wave_direction=0)
        wave_force = np.asarray(froude.values + diffraction.values)
        return {
            "omega": np.asarray(dataset["omega"].values, dtype=float).reshape(-1),
            "added_mass": np.asarray(dataset["added_mass"].values, dtype=float),
            "radiation_damping": np.asarray(dataset["radiation_damping"].values, dtype=float),
            "wave_force": wave_force,
        }
    finally:
        dataset.close()


def dominant_diagonal_dof(radiation_damping: np.ndarray) -> int:
    """Return the diagonal DOF with the largest average radiation damping."""

    damping = np.asarray(radiation_damping, dtype=float)
    diag = np.diagonal(damping, axis1=1, axis2=2)
    return int(np.argmax(np.mean(np.abs(diag), axis=0)))


def dominant_force_dof(wave_force: np.ndarray) -> int:
    """Return the force DOF with the largest average transfer-function magnitude."""

    force = np.asarray(wave_force)
    magnitude = np.mean(np.abs(force.reshape(force.shape[0], -1)), axis=0)
    return int(np.argmax(magnitude))


def plot_hydrodynamic_ab_comparison(
    path: str | Path,
    original_omega: np.ndarray,
    original_added_mass: np.ndarray,
    original_damping: np.ndarray,
    extended_omega: np.ndarray,
    extended_added_mass: np.ndarray,
    extended_damping: np.ndarray,
    *,
    dof_index: int | None = None,
) -> Path:
    """Plot selected A(omega) and B(omega) before/after extrapolation."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dof = dominant_diagonal_dof(original_damping) if dof_index is None else int(dof_index)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(extended_omega, extended_added_mass[:, dof, dof], color="#d62728", linewidth=1.1, label="extended")
    axes[0].plot(original_omega, original_added_mass[:, dof, dof], color="#111111", linewidth=1.4, label="original")
    axes[0].set_ylabel("A(omega)")
    axes[0].set_title(f"Hydrodynamic added mass and radiation damping, DOF {dof}")
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    axes[1].plot(extended_omega, extended_damping[:, dof, dof], color="#d62728", linewidth=1.1, label="extended")
    axes[1].plot(original_omega, original_damping[:, dof, dof], color="#111111", linewidth=1.4, label="original")
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("B(omega)")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def plot_excitation_force_extrapolation_comparison(
    path: str | Path,
    original_omega: np.ndarray,
    original_force: np.ndarray,
    extended_omega: np.ndarray,
    extended_force: np.ndarray,
    *,
    dof_index: int | None = None,
) -> Path:
    """Plot selected wave-excitation transfer-function magnitude."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dof = dominant_force_dof(original_force) if dof_index is None else int(dof_index)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(extended_omega, np.abs(extended_force.reshape(extended_force.shape[0], -1)[:, dof]), color="#d62728", linewidth=1.1, label="extended")
    ax.plot(original_omega, np.abs(original_force.reshape(original_force.shape[0], -1)[:, dof]), color="#111111", linewidth=1.4, label="original")
    ax.set_xlabel("Angular frequency (rad/s)")
    ax.set_ylabel("|F_ex(omega)|")
    ax.set_title(f"Excitation-force extrapolation, DOF {dof}")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def plot_radiation_kernel_norm(
    path: str | Path,
    time: np.ndarray,
    kernel: np.ndarray,
    *,
    title: str,
) -> Path:
    """Plot Frobenius norm and trace of a radiation kernel."""

    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(kernel.reshape(kernel.shape[0], -1), axis=1)
    trace = np.trace(kernel, axis1=1, axis2=2)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    axes[0].plot(time, norms, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("||K_r(t)||_F")
    axes[0].set_title(title)
    axes[0].grid(True, color="#dddddd", linewidth=0.7)
    axes[1].plot(time, trace, color="#9467bd", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("trace(K_r)")
    axes[1].grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output
