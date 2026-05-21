# RODM Full Regression Report

Date: 2026-05-09 08:56:21

## Scope

This report compares the refactored `solve_rodm_frequency_case()` output
against the existing `displacement_55mesh_300.npy` baseline for the
300 m x 60 m floating-body reference case.

Expected numerical-result change: none. Any nonzero difference should be
investigated before legacy scripts are redirected to the package solver.

## Runtime

- elapsed_seconds: `4.953`
- generated_response: `results\reference_case_300_rodm_generated.npy`
- hydro_reversed_elapsed_seconds: `4.477`
- hydro_reversed_response: `results\reference_case_300_rodm_hydro_reversed.npy`

## Default Solver vs Baseline

| Metric | Value |
| --- | ---: |
| shape_generated | `(3965, 1)` |
| shape_baseline | `(3965, 1)` |
| max_abs_error | `1.5771240988045787` |
| mean_abs_error | `0.24598219259348353` |
| l2_abs_error | `34.06180322136642` |
| l2_relative_error | `1.3203767908803576` |

## Default Solver Centerline Heave

| Metric | Value |
| --- | ---: |
| heave_len | `60` |
| heave_max_abs_error | `0.13407301974258234` |
| heave_rmse | `0.08924022041228792` |
| heave_generated_min | `0.7090060758386311` |
| heave_generated_max | `1.1809018748237416` |
| heave_generated_mean | `0.8114840772395009` |
| heave_baseline_min | `0.8166492461156476` |
| heave_baseline_max | `1.2525429563334871` |
| heave_baseline_mean | `0.896981533364751` |

## Default Solver vs External Curves

| Metric | Value |
| --- | ---: |
| generated_rmse_vs_exp300 | `0.07885618085858116` |
| generated_rmse_vs_fu_sim300 | `0.04545476944844251` |

## Hydrodynamic-Node-Reversed Candidate vs Baseline

This candidate reverses the 10 hydrodynamic node blocks before solving.
It is not the default legacy-equivalent path, but it matches the saved
baseline heave curve much more closely and may reflect the historical
notebook path that created `displacement_55mesh_300.npy`.

| Metric | Value |
| --- | ---: |
| shape_generated | `(3965, 1)` |
| shape_baseline | `(3965, 1)` |
| max_abs_error | `0.09216475671969848` |
| mean_abs_error | `0.015464057242424861` |
| l2_abs_error | `2.128506192132857` |
| l2_relative_error | `0.08250972965443046` |

## Hydrodynamic-Node-Reversed Centerline Heave

| Metric | Value |
| --- | ---: |
| heave_len | `60` |
| heave_max_abs_error | `0.003368025748696324` |
| heave_rmse | `0.001052959780126104` |
| heave_generated_min | `0.8170774052546647` |
| heave_generated_max | `1.2491749305847908` |
| heave_generated_mean | `0.8968750658850498` |
| heave_baseline_min | `0.8166492461156476` |
| heave_baseline_max | `1.2525429563334871` |
| heave_baseline_mean | `0.896981533364751` |

## Hydrodynamic-Node-Reversed vs External Curves

| Metric | Value |
| --- | ---: |
| generated_rmse_vs_exp300 | `0.0639975101923516` |
| generated_rmse_vs_fu_sim300 | `0.044886211767181924` |
