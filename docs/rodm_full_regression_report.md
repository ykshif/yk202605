# RODM Full Regression Report

Date: 2026-04-28 08:29:53

## Scope

This report compares the refactored `solve_rodm_frequency_case()` output
against the existing `displacement_55mesh_300.npy` baseline for the
300 m x 60 m floating-body reference case.

Expected numerical-result change: none. Any nonzero difference should be
investigated before legacy scripts are redirected to the package solver.

## Runtime

- elapsed_seconds: `13.407`
- generated_response: `results\reference_case_300_rodm_generated.npy`
- hydro_reversed_elapsed_seconds: `12.231`
- hydro_reversed_response: `results\reference_case_300_rodm_hydro_reversed.npy`

## Default Solver vs Baseline

| Metric | Value |
| --- | ---: |
| shape_generated | `(3965, 1)` |
| shape_baseline | `(3965, 1)` |
| max_abs_error | `1.577124150471191` |
| mean_abs_error | `0.24598219957354614` |
| l2_abs_error | `34.06180394246738` |
| l2_relative_error | `1.320376818833222` |

## Default Solver Centerline Heave

| Metric | Value |
| --- | ---: |
| heave_len | `60` |
| heave_max_abs_error | `0.1340730834715389` |
| heave_rmse | `0.08924026120495829` |
| heave_generated_min | `0.7090060782686021` |
| heave_generated_max | `1.180901829824625` |
| heave_generated_mean | `0.811484043457774` |
| heave_baseline_min | `0.8166492461156475` |
| heave_baseline_max | `1.2525429563334871` |
| heave_baseline_mean | `0.896981533364751` |

## Default Solver vs External Curves

| Metric | Value |
| --- | ---: |
| generated_rmse_vs_exp300 | `0.07885619177953455` |
| generated_rmse_vs_fu_sim300 | `0.04545481321986492` |

## Hydrodynamic-Node-Reversed Candidate vs Baseline

This candidate reverses the 10 hydrodynamic node blocks before solving.
It is not the default legacy-equivalent path, but it matches the saved
baseline heave curve much more closely and may reflect the historical
notebook path that created `displacement_55mesh_300.npy`.

| Metric | Value |
| --- | ---: |
| shape_generated | `(3965, 1)` |
| shape_baseline | `(3965, 1)` |
| max_abs_error | `0.09216475453142021` |
| mean_abs_error | `0.01546405987317927` |
| l2_abs_error | `2.128506207642789` |
| l2_relative_error | `0.08250973025565976` |

## Hydrodynamic-Node-Reversed Centerline Heave

| Metric | Value |
| --- | ---: |
| heave_len | `60` |
| heave_max_abs_error | `0.003368226666994767` |
| heave_rmse | `0.0010529882917823458` |
| heave_generated_min | `0.8170774454085562` |
| heave_generated_max | `1.2491747296664923` |
| heave_generated_mean | `0.8968750235622073` |
| heave_baseline_min | `0.8166492461156475` |
| heave_baseline_max | `1.2525429563334871` |
| heave_baseline_mean | `0.896981533364751` |

## Hydrodynamic-Node-Reversed vs External Curves

| Metric | Value |
| --- | ---: |
| generated_rmse_vs_exp300 | `0.06399745451451948` |
| generated_rmse_vs_fu_sim300 | `0.044886194123330454` |
