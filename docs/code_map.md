# Codebase Map

## 1. Current Repository Overview

This repository is a flat research-code workspace for offshore floating photovoltaic hydroelastic analysis. The current code supports frequency-domain hydroelastic response calculation, reduced-order dynamic modeling, SEREP/static-condensation reconstruction, hinge/interconnection studies, wind-wave coupled response, Abaqus boundary-condition generation, validation plotting, and an early PV power-generation notebook.

The repository is currently organized as loose files rather than a package:

- Python modules: numerical kernels, readers, wind loads, reduction methods, hinge utilities, plotting helpers.
- Jupyter notebooks: experiment history, validation studies, wind-wave studies, hinge studies, paper figures, and PV prototype work.
- Local artifacts: generated Abaqus `.inp` files and `.npy` displacement/response arrays.
- External data dependencies: most hydrodynamic, structural, and wind-coefficient inputs are referenced through hard-coded paths outside this folder, especially under `E:\phd\Code\DM-FEM2D\...`.

The project is best understood as a research prototype that has already identified the main physics modules, but has not yet separated source code, configuration, input data, outputs, and experiments.

## 2. Main Scripts And Their Functions

### `RODM_Wind_main.py`

Current closest thing to an executable integrated workflow.

Main responsibilities:

- Defines sea states, wind speeds, wind direction, and node count in `__main__`.
- Loads Capytaine hydrodynamic `.nc` data using `xarray` and `merge_complex_values`.
- Loads Abaqus mass and stiffness `.mtx` files through `DM_Reading`.
- Removes the 6th DOF and builds reduced structural matrices.
- Adds mooring stiffness.
- Builds JONSWAP wave spectrum.
- Builds wind damping and lumped wind forces.
- Solves the frequency-domain response for each omega.
- Reconstructs full displacement using reduction transformation matrices.
- Saves `.npy` displacement results.
- Provides simple response-spectrum and RMS contour plotting helpers.

### `DM_Method.py`

Higher-level reusable RODM workflows.

Main functions:

- `perform_RODM_reduce_order_model(...)`: SEREP-based reduced-order solve.
- `perform_expansion_and_solve(...)`: expansion-style solve and reconstruction.
- `calculate_initial_displacement(...)`: original full/insertion-style displacement calculation.
- `replace_master_with_global(...)`: patches reconstructed master-node values.

This module is a good candidate for becoming the first stable simulation API after refactoring.

### `SEREP.py`

Core reduction and reconstruction algorithms.

Main functions:

- `reduce_dofs(...)`
- `transform_mass_matrix(...)`
- `separate_dofs(...)`
- `SEREP(...)`
- `SEREP_Expansion(...)`
- `reduce_force_matrix_dofs(...)`
- `reorder_displacement_matrix(...)`
- `get_fem_spring_stiffness(...)`
- `calculate_hydrostatic_stiffness_matrix(...)`
- `static_condensation(...)`
- `dynamic_condensation(...)`
- `true_dynamic_condensation(...)`

This is one of the most important and riskiest numerical files.

### `DM_Assemble.py`

Matrix and force assembly plus frequency-domain solver.

Main functions:

- `insert_matrix(...)`
- `sparse_insert_matrix(...)`
- `extend_force_matrix(...)`
- `solve_frequency_domain(...)`
- `sparse_solve_frequency_domain(...)`
- `calculate_node_positions(...)`
- `calculate_2d_node_positions_descending(...)`

The central equation solved is:

```text
(-omega^2 M - i omega C + K) X = F
```

### `DM_Reading.py`

Readers for Abaqus structural matrices.

Main functions:

- `get_stiffness_matrix(...)`: dense matrix reader.
- `get_stiffness_csr_matrix(...)`: CSR conversion after dense construction.
- `get_stiffness_csr_matrix_optimized(...)`: sparse-first reader.
- `read_element_stiffness_matrix(...)`: reads a 24x24 element stiffness matrix from Abaqus-style matrix output.

### `DM_Windload.py`

Wind spectrum, wind coefficient, wind damping, and wind force model.

Main class:

- `WindLoad`

Main responsibilities:

- Adjust wind speed by height.
- Compute turbulence intensity.
- Compute API wind spectrum.
- Convert spectrum to amplitude.
- Read wind coefficient text files.
- Compute distributed and lumped wind forces.
- Compute wind damping matrices.

### `DM_Hinge.py`

Older/specific hinge connection workflow for a rectangular grid.

Main responsibilities:

- Finds interface node columns.
- Generates element connectivity across an interface.
- Removes element stiffness from global stiffness.
- Adds hinge coupling stiffness between node pairs.

### `RODM_complex_interconnection.py`

More general hinge/interconnection utilities for multi-module arrays.

Main responsibilities:

- Generate x-direction hinge pairs.
- Generate y-direction hinge pairs.
- Apply hinge stiffness blocks to a global stiffness matrix.
- Visualize module and hinge layout.

### `DM_Abaqus_inp.py`

Abaqus boundary-condition file generator.

Main responsibilities:

- Create node set definitions.
- Create static, real, and imaginary boundary conditions.
- Write generated `.inp` boundary-condition files.
- Modify boundary-condition DOF selections.

### `DM_forec_analysis.py`

Postprocessing for module internal force/stress analysis.

Main class:

- `ForceAnalysis`

Main responsibilities:

- Generate module node groups.
- Extract module displacements from global displacement.
- Compute module forces using element stiffness.
- Map module forces back to global nodes.
- Extract interface and boundary forces.
- Plot line curves, heatmaps, and 3D force surfaces.

### `DM_ShowNodes.py`

Reads Abaqus `.inp` geometry and plots FEA nodes/elements.

### `DM_Verify.py`

Hydrodynamic verification and experimental-data reading.

Main responsibilities:

- Load hydrodynamic `.nc`.
- Compute RAO with Capytaine.
- Plot RAO.
- Read two-column experimental data text files.

### `wave_spectrum.py`

Contains the JONSWAP wave spectrum function.

### `Pvlib.ipynb`

Prototype PV generation notebook using `pvlib`.

Current responsibilities:

- Define site location.
- Compute clear-sky irradiance.
- Compute cell temperature.
- Estimate DC power with PVWatts.

This is currently isolated from the hydroelastic response workflow.

## 3. Input-Output Data Flow

### Main Frequency-Domain Hydroelastic Flow

```text
Abaqus .mtx mass/stiffness
        |
        v
DM_Reading.get_stiffness_matrix
        |
        v
SEREP.reduce_dofs / transform_mass_matrix / separate_dofs
        |
        v
SEREP.SEREP or SEREP.static_condensation
        |
        v
Reduced structural M, K, T

Capytaine .nc hydrodynamic dataset
        |
        v
xarray.open_dataset + merge_complex_values
        |
        v
added_mass, radiation_damping, hydrostatic_stiffness,
Froude_Krylov_force, diffraction_force, omega
        |
        v
DOF reduction and force reduction

Wave spectrum / wind load model
        |
        v
frequency-dependent wave and wind forces
        |
        v
DM_Assemble.solve_frequency_domain
        |
        v
master displacement
        |
        v
T @ master_displacement
        |
        v
SEREP.reorder_displacement_matrix
        |
        v
global displacement
        |
        v
.npy results, plots, Abaqus boundary-condition .inp
```

### Current Local Artifacts

Local generated/input-like files:

- `Boundary_Conditions_Job-1.inp`
- `Boundary_Conditions_Job-1_3.inp`
- `Boundary_Conditions_Job-1_35.inp`
- `Boundary_Conditions_Job-1_3_6.inp`
- `displacement_1530mesh_300.npy`
- `displacement_33mesh_300.npy`
- `displacement_55mesh_300.npy`
- `dynamic_wl145_d180.npy`
- `Serep_wl145_d180.npy`
- `Static_wl145_d180.npy`

External expected files:

- Hydrodynamic `.nc`: Capytaine datasets.
- Structural `.mtx`: Abaqus mass/stiffness matrices.
- Element `.mtx`: submodule or hinge element stiffness matrices.
- Wind `.txt`: drag/lift coefficient files.
- Experimental `.txt`: validation curves.

## 4. Current Module Map

```text
RODM_Wind_main.py
    -> wave_spectrum.py
    -> DM_Windload.py
    -> DM_Reading.py
    -> DM_Assemble.py
    -> SEREP.py
    -> xarray / Capytaine merge_complex_values

DM_Method.py
    -> DM_ShowNodes.py
    -> DM_Reading.py
    -> DM_Assemble.py
    -> SEREP.py
    -> xarray / Capytaine merge_complex_values

SEREP.py
    -> numpy
    -> scipy.linalg.eigh / eig

DM_Assemble.py
    -> numpy
    -> scipy.sparse
    -> scipy.sparse.linalg.spsolve

DM_Reading.py
    -> numpy
    -> scipy.sparse.csr_matrix

DM_Windload.py
    -> numpy
    -> matplotlib

DM_Hinge.py
    -> numpy
    -> DM_Reading.py

RODM_complex_interconnection.py
    -> numpy
    -> matplotlib

DM_Abaqus_inp.py
    -> numpy
    -> re

DM_forec_analysis.py
    -> numpy
    -> matplotlib
    -> scipy.interpolate

DM_Verify.py
    -> matplotlib
    -> xarray
    -> Capytaine rao

DM_ShowNodes.py
    -> re
    -> matplotlib

Pvlib.ipynb
    -> pvlib
    -> pandas
    -> matplotlib
```

## 5. Problems In The Current Structure

### Flat Layout

All code, notebooks, generated files, and binary results live together in one directory. This makes it hard to distinguish source code from experiments and outputs.

### Hard-Coded Paths

Many scripts and notebooks reference absolute paths such as:

```text
E:\phd\Code\DM-FEM2D\...
```

This prevents portable execution and makes reproducibility difficult.

### Hard-Coded Model Parameters

Important physical and mesh parameters are embedded directly in scripts:

- `793`, `2121`
- `13`, `61`, `31`, `151`
- `424, 6, 10`
- removed DOF `[5]`
- `1e5`, `1e6`, `10e15`
- wind damping multiplier `5.9`
- stiffness scale `KR * 0.01`
- frequency spacing `0.01`
- reshape assumptions such as `(199, 793)`

These should become named configuration values.

### Duplicated Algorithms

The following appear both in modules and notebooks:

- SEREP workflow.
- Static/dynamic condensation.
- Hydrodynamic matrix extraction.
- Frequency-domain response loops.
- Hinge stiffness assembly.
- Plotting and validation routines.

### Fragile Global State

Some functions rely on variables not passed as arguments. For example, `run_simulation(...)` in `RODM_Wind_main.py` references `num_nodes` from outer/global scope.

### Ambiguous DOF Conventions

The code frequently removes `[5]`, meaning the 6th DOF in zero-based indexing. Some arrays use 6 DOFs per node, while many reduced workflows use 5 DOFs per node. This convention should be made explicit and tested.

### Dense Matrix Memory Risk

`DM_Reading.get_stiffness_matrix(...)` reads sparse Abaqus matrices into dense arrays. This is risky for larger models and may become a bottleneck for an integrated simulation platform.

### Notebook-Centered Workflow

Important calculations and decisions are embedded in notebooks, including old versions, copied versions, large outputs, and saved exceptions. This makes it difficult to know which workflow is authoritative.

### Encoding Issues

Several comments appear mojibake/garbled, likely due to encoding mismatch. This does not block execution but reduces maintainability.

## 6. Target Architecture

Recommended future package layout:

```text
rodm/
    __init__.py

    config/
        cases.py
        mesh.py
        paths.py
        dofs.py

    io/
        abaqus_mtx.py
        abaqus_inp.py
        capytaine_nc.py
        wind_coefficients.py
        results.py

    hydro/
        dataset.py
        forces.py
        hydrostatics.py

    structure/
        matrices.py
        mesh_nodes.py
        mooring.py
        hinge.py
        interconnection.py

    reduction/
        dofs.py
        serep.py
        condensation.py
        reconstruction.py

    loads/
        wave_spectrum.py
        wind.py
        combined.py

    solver/
        frequency_domain.py
        simulation.py

    pv/
        pv_model.py
        loss_model.py

    post/
        force_analysis.py
        plots.py
        validation.py
```

Recommended top-level project layout:

```text
configs/
    example_case.yaml
    wind_wave_case.yaml
    hinge_case.yaml

data/
    README.md
    raw/
    processed/

results/
    README.md

notebooks/
    archive/
    active/

scripts/
    run_frequency_case.py
    run_wind_wave_case.py
    run_hinge_case.py
    postprocess_validation.py

tests/
    test_dofs.py
    test_reconstruction.py
    test_frequency_solver.py
    test_matrix_readers.py
```

The target platform should expose a small number of stable user-facing workflows:

- run hydroelastic frequency-domain analysis;
- run wind-wave coupled response analysis;
- reconstruct full-field displacement;
- export Abaqus boundary conditions;
- compute internal force/stress postprocessing;
- estimate PV power loss from displacement, pitch/roll, orientation, and irradiance;
- later run optimization loops over layout, stiffness, mooring, hinge, or PV design variables.

## 7. Recommended Refactoring Order

1. Freeze one or two reference cases.

   Pick a small case and a current wind-wave case. Record input paths, output shapes, selected displacement norms, and expected plot behavior. This gives a baseline before moving code.

2. Create configuration files without changing math.

   Extract paths, mesh dimensions, DOF conventions, master-node definitions, frequency ranges, wind parameters, wave parameters, mooring stiffness, and hinge stiffness.

3. Separate source, notebooks, data, and results.

   Move only after baselines exist. Keep old notebooks in an archive folder and keep active notebooks thin.

4. Stabilize I/O modules.

   Build clean readers for Abaqus `.mtx`, Capytaine `.nc`, wind coefficient `.txt`, result `.npy`, and generated Abaqus `.inp`.

5. Add tests around DOF handling.

   First tests should cover `reduce_dofs`, `reduce_force_matrix_dofs`, `separate_dofs`, and `reorder_displacement_matrix`, because many downstream errors would come from indexing mistakes.

6. Extract the frequency-domain solver.

   Make `solve_frequency_domain(M, C, K, F, omega)` independent, documented, and tested against a tiny analytical system.

7. Refactor reduction algorithms.

   Move SEREP, static condensation, dynamic condensation, and reconstruction into a `reduction` package. Preserve existing formulas first; improve only after tests pass.

8. Refactor wind and wave loads.

   Keep JONSWAP and API wind spectrum separate from force assembly. Make frequency grid and spectral scaling explicit.

9. Refactor hinge/interconnection modeling.

   Merge `DM_Hinge.py`, notebook hinge code, and `RODM_complex_interconnection.py` into one tested interconnection module.

10. Refactor postprocessing.

   Move force/stress extraction and plotting out of notebooks into reusable functions. Keep paper-figure notebooks as thin visualization scripts.

11. Integrate PV generation.

   Turn `Pvlib.ipynb` into a `pv` module. Add a clear interface from hydroelastic response to panel orientation, irradiance loss, and power loss.

12. Build integrated simulation workflows.

   Add scripts or CLI commands for:

   - hydro-only response;
   - wind-wave response;
   - hinge/interconnection response;
   - stress postprocessing;
   - PV power-loss postprocessing;
   - future optimization studies.

