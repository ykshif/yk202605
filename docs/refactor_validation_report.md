# Refactor Validation Report

Date: 2026-04-27

## Scope

This refactor created a non-invasive package layer under `src/offshore_energy_sim`
and moved the 300 m x 60 m reference-case validation, plotting, and read-only
input checks into reusable modules.

No existing numerical solver, reduction algorithm, structural assembly routine,
source data file, or paper-reproduction script was changed. The legacy scripts
remain runnable:

- `scripts/verify_reference_case_300.py`
- `scripts/plot_reference_case_300.py`
- `scripts/validate_reference_case_300_inputs.py`
- `scripts/validate_reduction_solver_kernels.py`
- `scripts/validate_structure_connectors.py`
- `scripts/validate_rodm_case_orchestration.py`
- `scripts/validate_environment_load_power_strength.py`
- `scripts/run_reference_case_300_rodm_compare.py`
- `scripts/compare_legacy_and_packaged_rodm.py`
- `scripts/investigate_reference_case_300_variants.py`

Expected numerical-result change: none.

## New Package Structure

```text
src/offshore_energy_sim/
|-- core/
|   |-- cases.py
|   |-- dependencies.py
|   `-- paths.py
|-- environment/
|   |-- spectra.py
|   `-- waves.py
|-- geometry/
|   `-- floating_body.py
|-- hydrodynamics/
|   `-- netcdf.py
|-- postprocess/
|   |-- metrics.py
|   |-- plots.py
|   `-- reference_case_300.py
|-- reduction/
|   |-- dofs.py
|   `-- modal.py
|-- response/
|   |-- retained_dofs.py
|   `-- spectral.py
|-- solver/
|   |-- frequency_domain.py
|   `-- rodm_frequency.py
|-- structure/
|   |-- assembly.py
|   |-- connectors.py
|   |-- hinges.py
|   `-- matrix_io.py
|-- loads/
|   |-- vector_mapping.py
|   `-- wind.py
|-- power/
|   `-- pv.py
|-- strength/
|   `-- internal_forces.py
`-- utils/
    `-- hashing.py
```

The remaining major placeholder is `optimization`; other layers now have
initial package interfaces but are not yet wired into legacy paper-reproduction
scripts.

## Functional Mapping

| Current package module | Responsibility |
| --- | --- |
| `core.cases` | Standard dataclasses for master-node rules, structural matrix paths, and RODM frequency cases. |
| `core.dependencies` | Optional dependency checks for heavy solver paths. |
| `geometry.floating_body` | Rectangular floating-body geometry metadata for reference cases. |
| `environment.waves` | Regular-wave case descriptor. |
| `environment.spectra` | JONSWAP wave spectrum, API wind spectrum, and spectrum-to-amplitude helpers. |
| `hydrodynamics.netcdf` | Optional xarray/Capytaine NetCDF loader and file summary. |
| `structure.matrix_io` | Abaqus-style `.mtx` scanner and dense matrix reader with explicit DOF indexing. |
| `structure.assembly` | Global DOF indexing, local-to-global matrix insertion, and node-position helpers. |
| `structure.connectors` | Generic two-node connector stiffness assembly with `+KC/-KC` block coupling. |
| `structure.hinges` | Hinge column nodes, element removal, hinge coupling matrix, and element stiffness block reader. |
| `reduction.dofs` | Matrix/force DOF removal, master/slave DOF split, displacement reordering, and master DOF replacement. |
| `reduction.modal` | Mass-matrix transform and optional SciPy-based SEREP/SEREP expansion formulas. |
| `solver.frequency_domain` | Frequency-domain MCK dynamic stiffness and displacement solve. |
| `solver.rodm_frequency` | Case-level RODM frequency-domain orchestration matching the legacy `DM_Method` workflow. |
| `loads.vector_mapping` | Mapping local nodal force blocks into full global force vectors. |
| `loads.wind` | Distributed wind force/damping and submodule coefficient splitting. |
| `response.retained_dofs` | Retained-DOF response vector extraction with explicit node and DOF indexing. |
| `response.spectral` | Response spectrum and RMS helpers. |
| `strength.internal_forces` | Module node generation, module displacement extraction, internal force mapping, and interface moment extraction. |
| `power.pv` | Lightweight PV DC power and cosine tilt-loss helpers. |
| `postprocess.metrics` | Validation metrics such as RMSE. |
| `postprocess.plots` | Reusable heave RAO comparison plotting. |
| `postprocess.reference_case_300` | Case-specific paths, hashes, metrics, extraction, input summary, verification, and plotting. |
| `utils.hashing` | SHA-256 file hashing for reproducibility checks. |

## Input Validation

Command:

```powershell
python scripts\validate_reference_case_300_inputs.py
```

Result: passed.

Hydrodynamic input:

| Item | Value |
| --- | --- |
| file | `E:\phd\Code\DM-FEM2D\HydrodynamicData\Yoga\DM10_300_direction0.nc` |
| sha256 | `D2414083E634B958139C5A4203BFD2C7AFA1782D34D4A80F0F12E669BD8EEEC9` |
| xarray_available | `False` in the current Python environment |
| capytaine_available | `False` in the current Python environment |

Structural inputs:

| Matrix | Shape | Stored entries | Symmetric entries estimate | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Mass | `(4758, 4758)` | `19518` | `34278` | `FDB09EB5149417A0EE3BAB01827F128EF6F1D2A82A0A56709A65422E8A45009B` |
| Stiffness | `(4758, 4758)` | `51944` | `99130` | `4D7B48381323F35210A38469A4F8BC81533FFC57473682BF2108E2A69C5566AA` |

The structural matrix shape corresponds to `793 nodes * 6 full DOFs`.

## Baseline Verification

Command:

```powershell
python scripts\verify_reference_case_300.py
```

Result: passed.

Key metrics:

| Metric | Value |
| --- | ---: |
| response_shape | `(3965, 1)` |
| response_dtype | `complex128` |
| heave_len | `60` |
| heave_abs_min | `0.8166492461156475` |
| heave_abs_max | `1.2525429563334871` |
| heave_abs_mean | `0.896981533364751` |
| heave_abs_l2 | `7.000919304253492` |
| rmse_vs_exp300 | `0.06367482251124734` |
| rmse_vs_fu_sim300 | `0.04488934895346538` |

The current result, hydrodynamic file, structural matrices, experiment data, and
Fu et al. comparison data all matched their documented SHA-256 hashes.

## Reduction And Solver Kernel Validation

Command:

```powershell
python scripts\validate_reduction_solver_kernels.py
```

Result: passed.

Validated kernels:

- `reduce_matrix_dofs`: equivalent node-major local DOF deletion.
- `reduce_force_dofs`: equivalent flattened force-vector DOF deletion.
- `transform_mass_matrix`: same consistent/lumped mass blending formula.
- `separate_master_slave_dofs`: same one-based master node to global DOF mapping.
- `reorder_displacement_to_natural_order`: same reverse-master reordering convention.
- `replace_master_dofs_in_global_response`: same master response replacement convention.
- `extend_force_vector_to_nodes`: same `(1, total_nodes * dofs_per_node)` force expansion.
- `solve_frequency_domain`: residual check for
  `(-omega**2*M - 1j*omega*C + K) X = F.T`.

Expected numerical-result change: none. This step only added named package
interfaces around formulas already present in `SEREP.py`, `DM_Assemble.py`, and
`DM_Method.py`.

## Structure Connector Validation

Command:

```powershell
python scripts\validate_structure_connectors.py
```

Result: passed.

Validated kernels:

- `node_dof_indices`: one-based node IDs to node-major global DOF indices.
- `assemble_local_matrix`: local matrix insertion into a zero global matrix.
- `calculate_node_positions`: descending control-point nodes.
- `calculate_2d_node_positions_descending`: descending 2D module node layout.
- `calculate_column_node_indices`: column node extraction for hinge lines.
- `generate_column_elements`: four-node elements between adjacent columns.
- `remove_element_stiffness_in_place`: equivalent element stiffness subtraction.
- `add_two_node_coupling_in_place`: generic two-node `+KC/-KC` connector assembly.
- `add_hinge_connections_in_place`: equivalent hinge assembly with released local DOF index 4.

Expected numerical-result change: none. This step only added package interfaces
around `DM_Hinge.py` and structural assembly helpers in `DM_Assemble.py`.

## RODM Case Orchestration Validation

Command:

```powershell
python scripts\validate_rodm_case_orchestration.py
```

Result: passed.

Validated configuration:

- `case_id`: `reference_case_300`
- master nodes: `[424, 418, 412, 406, 400, 394, 388, 382, 376, 370]`
- retained full response DOFs: `793 nodes * 5 DOFs = 3965`
- reduced hydrodynamic DOFs: `10 nodes * 5 DOFs = 50`
- solver entry point: `solve_rodm_frequency_case`

Current optional dependency status:

- missing: `xarray`, `capytaine`, `scipy`

Using the Anaconda base environment at `D:\Users\KKKKK\anaconda3`, these
dependencies are available through:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python ...
```

Numerical-result expectation: unchanged for the default packaged solver because
it follows the existing `DM_Method.perform_RODM_reduce_order_model` sequence.

## Full RODM Regression In Conda Base

Commands:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_reference_case_300_rodm_compare.py
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\compare_legacy_and_packaged_rodm.py
```

Result: the default packaged solver is exactly equivalent to the current legacy
`DM_Method.perform_RODM_reduce_order_model`, but this legacy path does not
exactly reproduce the saved `displacement_55mesh_300.npy` baseline.

Default packaged solver vs legacy `DM_Method`:

| Metric | Value |
| --- | ---: |
| max_abs_error | `0.0` |
| l2_relative_error | `0.0` |

Default packaged solver vs saved baseline:

| Metric | Value |
| --- | ---: |
| shape | `(3965, 1)` |
| max_abs_error | `1.577124150471191` |
| l2_relative_error | `1.320376818833222` |
| centerline_heave_rmse | `0.08924026120495829` |

Hydrodynamic-node-reversed candidate vs saved baseline:

| Metric | Value |
| --- | ---: |
| shape | `(3965, 1)` |
| max_abs_error | `0.09216475453142021` |
| l2_relative_error | `0.08250973025565976` |
| centerline_heave_rmse | `0.0010529882917823458` |
| generated_rmse_vs_exp300 | `0.06399745451451948` |
| generated_rmse_vs_fu_sim300 | `0.044886194123330454` |

Interpretation: the saved baseline was likely produced by a notebook variant
that reversed the 10 hydrodynamic node blocks before solving. This is supported
by the much closer centerline heave agreement, but the full response still has
nonzero differences and should not be treated as proven identical yet.

Generated outputs:

- `results/reference_case_300_rodm_generated.npy`
- `results/reference_case_300_rodm_hydro_reversed.npy`
- `results/reference_case_300_legacy_dm_method.npy`
- `docs/rodm_full_regression_report.md`

## Environment, Load, Power, And Strength Validation

Command:

```powershell
python scripts\validate_environment_load_power_strength.py
```

Result: passed.

Validated kernels:

- `jonswap_spectrum`: equivalent to `wave_spectrum.jonswap`.
- `api_wind_spectrum`, `wind_speed_power_law`, `turbulence_intensity_api`,
  and `amplitude_from_spectrum`: equivalent to the scalar formulas in
  `DM_Windload.py`.
- `distributed_wind_force` and `distributed_wind_damping`: wind force/damping
  insertion into selected global DOFs.
- `split_submodule_coefficients`: submodule wind coefficient splitting logic.
- `generate_1d_module_nodes`, `extract_module_displacements`,
  `compute_module_forces`, `map_module_forces_to_global_nodes`, and
  `middle_interface_moment_per_width`: module force-analysis kernels migrated
  from `DM_forec_analysis.py`.
- `dc_power_from_irradiance`, `power_with_tilt_loss`, and
  `relative_power_loss`: initial PV power-loss interface.
- `response_spectrum_from_amplitude` and `rms_from_spectrum`: response
  spectrum helper interface.

Expected numerical-result change: none for migrated wave, wind, and
force-analysis formulas. The PV helpers are new isolated utilities and are not
connected to any legacy baseline result yet.

## Comparison Figure

Command:

```powershell
python scripts\plot_reference_case_300.py
```

Generated files:

- `figures/reference_case_300_heave_vs_experiment.png`
- `figures/reference_case_300_heave_vs_experiment.pdf`

The figure compares:

- Present RODM result extracted from `displacement_55mesh_300.npy`
- Experimental heave RAO from `exp_300.txt`
- Fu et al. (2007) simulation curve from `fu_sim300.txt`

## Numerical Equivalence Notes

The response extraction remains equivalent to the original script:

```python
midline = response[367 * 5 - 5 : 427 * 5 - 5, :]
heave = abs(midline[2::5, 0])
```

The refactored helper expresses the same indexing with named parameters:

```python
retained_node_dof_series(
    response,
    start_node_one_based=367,
    stop_node_one_based=427,
    retained_dofs_per_node=5,
    dof_index_zero_based=2,
    column=0,
)
```

Physical meaning: vertical heave response amplitude operator along the
validation centerline of the 300 m x 60 m floating body.

## Recommended Next Refactoring Step

The next low-risk step is to migrate orchestration around the existing kernels
while preserving legacy scripts:

1. `hydrodynamics`: after installing `xarray` and `capytaine`, add real dataset
   dimension/unit checks for added mass, radiation damping, hydrostatic
   stiffness, and wave excitation forces.
2. `solver`: identify the exact notebook path that created
   `displacement_55mesh_300.npy`, with special attention to hydrodynamic node
   order reversal and any master-DOF replacement.
3. `postprocess`: add more validation cases using the same verification pattern.
4. `power`: isolate PV power-loss calculations after displacement/attitude
   outputs have stable interfaces.

Each migration should keep a legacy script runnable and add a verification
command that states whether numerical results are expected to change.
# Workflow Platformization Update

Date: 2026-04-28

The previously staged RODM refactor has been applied to the formal repository.
The workflow layer now includes:

- `src/offshore_energy_sim/core/workflow.py` for standard output paths and `metrics.json`;
- `configs/templates/rodm_frequency_case.yaml` as a reusable case template;
- `configs/reference_case_300_hydro_reversed.yaml` as an explicit provenance variant;
- `scripts/run_rodm_case_from_config.py` for generic config-driven solving;
- `scripts/run_reference_case_300_workflow.py` for run/validate/plot/report;
- `scripts/validate_configured_variants.py` for explicit variant regression checks.

Standard output layout:

```text
results/<case_id>/
├── response.npy
├── metrics.json
├── report.md
├── figures/
└── logs/
```

Explicit variants use:

```text
results/<case_id>/variants/<variant_id>/
```

Latest numerical checks:

| Check | max_abs_error | l2_relative_error |
| --- | ---: | ---: |
| standard workflow default vs pre-refactor output | `0.0` | `0.0` |
| standard workflow hydro-reversed vs pre-refactor output | `0.0` | `0.0` |
| configured hydro-reversed variant vs candidate output | `0.0` | `0.0` |

Expected numerical change: none. Verified numerical change: none.

# Regular Wave Batch Validation Update

Date: 2026-04-28

The workflow layer has been used to run regular-wave validations for five
wavelengths:

```text
60 m, 120 m, 180 m, 240 m, 300 m
```

Entry point:

```text
scripts/run_regular_wave_batch_validation.py
```

Outputs:

```text
results/regular_wave_batch/
docs/regular_wave_batch_validation_report.md
```

RMSE summary against experiment:

| Wavelength (m) | Default RODM | Hydro-node-reversed |
| ---: | ---: | ---: |
| 60 | `0.03324548299186851` | `0.15525762547152588` |
| 120 | `0.029861016336435613` | `0.3457601513256085` |
| 180 | `0.03600375044035902` | `0.2787032253584788` |
| 240 | `0.07494131232813647` | `0.176705981792237` |
| 300 | `0.07885619177953455` | `0.06399745451451948` |

Interpretation:

- For 60 m, 120 m, 180 m, and 240 m, the default current-legacy-equivalent
  RODM path is closer to experiment.
- For 300 m, the explicit hydrodynamic-node-reversed candidate is closer to
  experiment, consistent with the earlier 300 m provenance investigation.

Expected numerical change from this batch validation script: none. It only
runs existing solver paths, computes metrics, and writes plots/reports.
