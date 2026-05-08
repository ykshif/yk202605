# MacBook Setup

Date: 2026-04-28

## Recommended Environment

Use the conda environment file at the repository root:

```bash
conda env create -f environment-mac.yml
conda activate offshore-energy-sim
```

If you use Apple Silicon, Miniforge or Mambaforge is recommended because it uses
`conda-forge` cleanly on `osx-arm64`.

The Mac environment file intentionally uses `conda-forge` only. This avoids the
non-interactive Terms-of-Service prompt now required by Anaconda `defaults`
channels and keeps dependency resolution aligned with the recommended Apple
Silicon stack. The local conda installation should set channel priority in
`.condarc`; `conda env create` ignores `channel_priority` if it is placed inside
an environment YAML file.

On Apple Silicon, `capytaine` may not be available from conda-forge as a native
`osx-arm64` conda package. The environment file therefore installs the rest of
the scientific stack with conda and leaves `capytaine` in the pip section.

The default Mac environment is kept lean enough for numerical validation.
Notebook and legacy visualization tools can be installed later if needed:

```bash
conda install -n offshore-energy-sim -c conda-forge jupyterlab ipykernel vtk
```

## Fallback Pip Environment

The pip fallback is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For this repository, conda is preferred because `capytaine`, `vtk`, and NetCDF
packages are more reliable through `conda-forge`.

## Data Path Migration

The current benchmark configs still reference Windows data paths such as:

```text
E:\phd\Code\DM-FEM2D\...
```

After copying the data to your MacBook, update the YAML files under `configs/`
to macOS paths, for example:

```text
/Users/<your-name>/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_300_direction0.nc
```

The most important files for the 300 m reference case are:

```text
DM10_300_direction0.nc
JobMesh5_5_MASS1.mtx
JobMesh5_5_STIF1.mtx
exp_300.txt
fu_sim300.txt
displacement_55mesh_300.npy
```

Several validation scripts now also support a Mac/Linux data-root override:

```bash
export RODM_DM_FEM_ROOT="/Users/<your-name>/data/DM-FEM2D"
```

After setting this variable, scripts such as
`scripts/validate_hinge_model.py`, `scripts/run_hinge_abaqus_benchmark.py`,
`scripts/run_yoon_hinge_response_validation.py`, and
`scripts/run_regular_wave_batch_validation.py` will look for external
hydrodynamic, structural, and comparison files under that root instead of the
default historical Windows path.

Yoon hinge comparison helper directories can be overridden separately when they
are copied to macOS:

```bash
export RODM_YOON_REFERENCE_DIR="/path/to/Yoon et al. 数值结果"
export RODM_YOON_HINGE_DIR="/path/to/FEM_Reducev2/Hinge"
export RODM_YOON_PDF_DIR="/path/to/RODM_AD/Hige"
```

## Environment Check

After activating the environment, run:

```bash
python scripts/check_environment.py
```

This checks core scientific dependencies and reports optional packages used by
legacy visualization and PV workflows.

## Basic Smoke Tests

Run these from the repository root:

```bash
python scripts/validate_reduction_solver_kernels.py
python scripts/validate_environment_load_power_strength.py
python scripts/validate_config_driven_reference_case.py
```

The full 300 m workflow requires the external hydrodynamic, structural, and
comparison data paths to exist on the MacBook:

```bash
python scripts/run_reference_case_300_workflow.py
```

Expected numerical-result change from moving to MacBook: none, assuming the same
input data files and compatible dependency versions are used.
