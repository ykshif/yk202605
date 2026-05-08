# Baseline Provenance

Date: 2026-04-28

## Question

The package solver can run the 300 m x 60 m RODM case in the Anaconda `base` environment, but the default solver output does not exactly match the saved `displacement_55mesh_300.npy` file. This document records the provenance investigation.

## Confirmed Facts

The default package solver is exactly equivalent to the current legacy function `DM_Method.perform_RODM_reduce_order_model`:

| Comparison | max_abs_error | l2_relative_error |
| --- | ---: | ---: |
| packaged default vs legacy `DM_Method` | `0.0` | `0.0` |

However, both differ from `displacement_55mesh_300.npy`:

| Comparison | max_abs_error | l2_relative_error | centerline_heave_rmse |
| --- | ---: | ---: | ---: |
| packaged default vs saved baseline | `1.577124150471191` | `1.320376818833222` | `0.08924026120495829` |

## Closest Candidate Path

The closest tested candidate reverses the 10 hydrodynamic node blocks before assembling the frequency-domain solve. It preserves local DOF order inside each node block.

| Comparison | max_abs_error | l2_relative_error | centerline_heave_rmse |
| --- | ---: | ---: | ---: |
| hydrodynamic-node-reversed candidate vs saved baseline | `0.09216475453142021` | `0.08250973025565976` | `0.0010529882917823458` |

This candidate also reproduces the external comparison metrics closely:

| Metric | Value |
| --- | ---: |
| generated_rmse_vs_exp300 | `0.06399745451451948` |
| generated_rmse_vs_fu_sim300 | `0.044886194123330454` |

The saved baseline metrics are:

| Metric | Value |
| --- | ---: |
| baseline_rmse_vs_exp300 | `0.06367482251124734` |
| baseline_rmse_vs_fu_sim300 | `0.04488934895346538` |

## Notebook Evidence

Notebook scans show repeated historical experiments with reversed ordering:

- `FEM_Reduce_v11.ipynb` contains an active force reversal: `F_w = F_w.reshape(10,5)[::-1].reshape(1,50)`.
- `RODM_Static_checkmesh.ipynb` contains active master displacement reversal followed by master DOF replacement: `master_displacement = master_displacement.reshape(len(master_nodes), 5)[::-1].reshape(5 * len(master_nodes), 1)`.
- `RODM_Static_checkmesh.ipynb` loads `E:\phd\Code\DM-FEM2D\FEM_Reduce\displacement_55mesh_300.npy` for comparison, but the scanned repository notebooks do not show a direct `np.save` for `displacement_55mesh_300.npy`.
- Several notebooks contain commented or active variants of `[::-1]` applied to wave force, master displacement, heave plotting, or module ordering.

## Interpretation

The saved `displacement_55mesh_300.npy` was likely produced by a historical notebook variant rather than the current `DM_Method.perform_RODM_reduce_order_model` function. The strongest numerical clue is the hydrodynamic-node-reversed candidate, which nearly matches the saved centerline heave curve.

The full response is not yet exactly identical, so this should be treated as a high-confidence lead rather than a proven reconstruction.

## Baseline Recommendation

For platform development, keep two explicit baselines:

1. `legacy_dm_method`: exact reproduction of the current root-level `DM_Method.perform_RODM_reduce_order_model`.
2. `saved_reference_300`: the published/experiment-comparison baseline stored in `displacement_55mesh_300.npy`.

Do not silently change the default solver to reverse hydrodynamic node order. Keep `reverse_hydrodynamic_node_order` as an explicit case option until the original notebook provenance is proven.
