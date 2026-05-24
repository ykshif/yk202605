# WEC-Sim-like Multi-Sea-State Validation - 2026-05-22

This note records the next validation step for the external RODM
WEC-Sim-like time-domain adapter. The run follows the requested scope:

```text
1. multi-sea-state validation
2. mooring module unchanged
3. continued state-space validation
4. long lightweight screening
5. continued comparison output
6. extension interfaces preserved
```

No RODM frequency-domain core files were modified, and the mooring adapter was
not changed. The existing four-corner spring provider was reused only through
the public adapter interface.

## Added Script

```text
scripts/validate_wecsim_like_multi_sea_state.py
```

The script orchestrates repeated calls to:

```text
solve_rodm_wecsim_like_time_domain(...)
```

It does not introduce a new solver. It only organizes sea states, runs direct
Cummins and ERA state-space comparisons, writes summary metrics, and performs a
longer state-space-only screening case.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py `
  --output-root results\time_domain\wecsim_like_multi_sea_state_validation `
  --hs-values 0.5,1.0 `
  --target-omega-values 0.35,0.4157,0.55 `
  --seeds 1 `
  --cycles 20 `
  --steps-per-cycle 40 `
  --memory-cycles 2 `
  --long-cycles 120 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

Input hydrodynamics:

```text
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
```

The short validation cases use direct Cummins convolution as the reference and
ERA-240 as the WEC-Sim-like state-space radiation option.

## Short Multi-Sea-State Results

| Case | Tp (s) | Samples | Master Error | Memory Error | Heave L2 Error | Heave RMS Error | ERA Fit Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hs0.5, omega 0.35 | 17.952 | 801 | 9.783e-3 | 2.520e-2 | 9.659e-3 | 3.049e-3 | 8.969e-3 |
| Hs0.5, omega 0.4157 | 15.115 | 801 | 9.242e-3 | 1.881e-2 | 9.174e-3 | 3.276e-3 | 6.640e-3 |
| Hs0.5, omega 0.55 | 11.529 | 801 | 1.080e-2 | 2.086e-2 | 1.039e-2 | 4.816e-3 | 1.176e-3 |
| Hs1.0, omega 0.35 | 17.952 | 801 | 9.783e-3 | 2.520e-2 | 9.659e-3 | 3.049e-3 | 8.969e-3 |
| Hs1.0, omega 0.4157 | 15.115 | 801 | 9.242e-3 | 1.881e-2 | 9.174e-3 | 3.276e-3 | 6.640e-3 |
| Hs1.0, omega 0.55 | 11.529 | 801 | 1.080e-2 | 2.086e-2 | 1.039e-2 | 4.816e-3 | 1.176e-3 |

Summary:

```text
short cases = 6
max state/direct master displacement error = 1.080e-2
max state/direct memory force error = 2.520e-2
max state/direct centerline heave RMS error = 4.816e-3
```

The identical relative errors for `Hs = 0.5 m` and `Hs = 1.0 m` are expected
because the current model is linear and the same random phase seed was used.
Absolute RMS levels scale with wave height, while relative state/direct errors
remain nearly unchanged.

## Long Lightweight Screening

The long screening case used only the ERA state-space radiation path:

```text
Hs = 1.0 m
omega_peak = 0.4157 rad/s
Tp = 15.1147 s
cycles = 120
duration = 1813.765 s
time samples = 4801
time step = 0.377868 s
```

No large arrays were saved. The output mode was:

```text
metrics_and_figures_only_no_large_arrays_saved
```

Long-run metrics:

```text
state master displacement RMS norm = 5.609e-1
state master velocity RMS norm = 2.407e-1
state memory force RMS norm = 5.612e6
centerline heave RMS mean = 1.763e-1
reconstructed Hs = 0.9996 m
```

Windowed checks over six time windows:

```text
wave Hs windows = [1.138, 1.021, 0.826, 1.071, 0.977, 0.935] m
mean centerline heave RMS windows = [0.204, 0.180, 0.148, 0.186, 0.169, 0.165]
```

The long result shows no obvious numerical blow-up in the state-space path. The
window-to-window variation is dominated by finite-duration random wave
realization statistics.

## Output Artifacts

```text
results/time_domain/wecsim_like_multi_sea_state_validation/multi_sea_state_metrics.json
results/time_domain/wecsim_like_multi_sea_state_validation/multi_sea_state_summary.csv
results/time_domain/wecsim_like_multi_sea_state_validation/report.md
results/time_domain/wecsim_like_multi_sea_state_validation/figures/state_space_direct_error_matrix.png
results/time_domain/wecsim_like_multi_sea_state_validation/figures/centerline_heave_rms_summary.png
results/time_domain/wecsim_like_multi_sea_state_validation/figures/long_lightweight_window_metrics.png
```

## Test Result

```text
79 passed in 2.64s
```

## Interpretation

The state-space model remains stable over the tested sea-state range and stays
close to direct Cummins convolution for hydroelastic heave observables. The
largest short-case centerline heave RMS error is below `0.5%`, and the largest
full master displacement error is about `1.1%`.

This supports continuing with ERA state-space as the practical radiation model
for longer RODM time-series studies, while keeping direct Cummins as the
reference validation path.

## Next Suggested Validation

The next validation should expand in two directions without changing the core
architecture:

```text
1. add more random seeds at the same Hs/Tp points;
2. add a wider Tp grid near the low-frequency edge and high-frequency tail;
3. keep direct Cummins comparison on short windows;
4. use state-space-only lightweight runs for long-duration screening;
5. add frequency-domain RMS comparison once the sea-state grid is fixed.
```
