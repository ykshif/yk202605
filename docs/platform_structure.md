# Platform Structure

Date: 2026-04-27

This document describes the current refactored package structure. Legacy
paper-reproduction scripts remain at the repository root and should stay
runnable during migration.

## Package Layers

```text
src/offshore_energy_sim/
|-- core/              case models, optional dependencies, shared paths
|-- geometry/          floating-body and mesh metadata
|-- environment/       waves, wave spectra, wind spectra
|-- hydrodynamics/     NetCDF hydrodynamic input adapters
|-- structure/         structural matrices, assembly, connectors, hinges
|-- reduction/         DOF reduction, SEREP/RODM transforms
|-- solver/            frequency-domain solver and case orchestration
|-- loads/             wave/wind/load-vector mapping
|-- response/          response extraction and spectral postprocessing
|-- strength/          module internal force and interface force helpers
|-- power/             PV power generation and loss helpers
|-- postprocess/       validation metrics, plots, reference cases
|-- optimization/      reserved for future optimization workflows
`-- utils/             hashing and general utilities
```

## Current Stable Validation Commands

```powershell
python scripts\validate_reference_case_300_inputs.py
python scripts\verify_reference_case_300.py
python scripts\plot_reference_case_300.py
python scripts\validate_reduction_solver_kernels.py
python scripts\validate_structure_connectors.py
python scripts\validate_rodm_case_orchestration.py
python scripts\validate_environment_load_power_strength.py
```

For dependency-enabled full RODM regression, use the Anaconda base environment:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_reference_case_300_rodm_compare.py
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\compare_legacy_and_packaged_rodm.py
```

## Numerical Result Expectations

The current refactor is interface-only for existing numerical workflows. It
does not change the baseline 300 m x 60 m displacement result or any source
data file.

The 300 m baseline still verifies:

- `response_shape = (3965, 1)`
- `rmse_vs_exp300 = 0.06367482251124734`
- `rmse_vs_fu_sim300 = 0.04488934895346538`

## Protected Legacy Areas

These files remain the source of paper reproduction behavior until a full
dependency-enabled regression is complete:

- `DM_Method.py`
- `SEREP.py`
- `DM_Assemble.py`
- `DM_Hinge.py`
- `DM_Windload.py`
- `DM_forec_analysis.py`
- `RODM_Wind_main.py`
- notebooks that reproduce published results

## Next Migration Boundary

The dependency-enabled full solve has now been run in `base`. The default
packaged solver exactly matches current `DM_Method.perform_RODM_reduce_order_model`,
but that path does not exactly match the saved `displacement_55mesh_300.npy`.

The next major boundary is historical baseline provenance:

1. Locate the notebook cell or script variant that created
   `displacement_55mesh_300.npy`.
2. Verify whether it reversed hydrodynamic node blocks before solving.
3. Decide whether the platform baseline should be the legacy `DM_Method` path
   or the saved `.npy` reference path.
4. Only after numerical equivalence is confirmed should legacy scripts begin to
   call the new package functions directly.
# Current Workflow Layer

Date: 2026-04-28

The platform now has a first standardized case workflow:

```text
core/config.py                 # YAML config loading
core/workflow.py               # standard output paths and metrics.json
hydrodynamics/frequency.py     # reduced frequency-domain hydrodynamic terms
structure/rodm_reduction.py    # retained-DOF structural matrices and SEREP terms
solver/rodm_frequency.py       # RODM frequency-domain orchestration
response/reconstruction.py     # global retained response reconstruction
postprocess/validation.py      # reusable response/curve metrics
postprocess/workflow_report.py # Markdown workflow reports
```

Primary entrypoints:

```text
scripts/run_rodm_case_from_config.py
scripts/run_reference_case_300_workflow.py
scripts/run_refactor_regression_suite.py
```

The standard output directory is:

```text
results/<case_id>/
```

with explicit variants under:

```text
results/<case_id>/variants/<variant_id>/
```

This layer is intended to be reused for later wind, floating PV, strength, and
optimization workflows without changing the underlying RODM numerical kernels.
