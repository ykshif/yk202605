# Spectrum Frequency-Time Response Comparison

Date: 2026-05-22

## Purpose

For a wave-spectrum sea state, compare:

```text
frequency-domain response spectrum:
  S_z(omega) = |RAO_z(omega)|^2 S_eta(omega)

time-domain response spectrum:
  fit z(t) = sum_j Re(Z_j exp(-i omega_j t))
  S_z,time(omega_j) = 0.5 |Z_j|^2 / Delta omega_j
```

The RMS comparison is:

```text
frequency RMS = sqrt(integral S_z(omega) d omega)
time fit RMS = sqrt(0.5 sum_j |Z_j|^2)
time series RMS = RMS(z(t) after startup discard)
```

## Script

```text
scripts/compare_spectrum_frequency_time_response.py
```

Default case:

```text
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/time_domain_extrapolated
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\compare_spectrum_frequency_time_response.py
```

## Extrapolated 142-Frequency Case

Output:

```text
results/time_domain/spectrum_frequency_time_comparison/frequency_time_motion_spectrum_metrics.json
results/time_domain/spectrum_frequency_time_comparison/centerline_rms_frequency_vs_time.csv
results/time_domain/spectrum_frequency_time_comparison/figures/motion_spectrum_frequency_vs_time.png
results/time_domain/spectrum_frequency_time_comparison/figures/centerline_rms_frequency_vs_time.png
results/time_domain/spectrum_frequency_time_comparison/figures/wave_and_midpoint_motion_spectrum.png
```

Metrics:

```text
component_count = 142
frequency_vs_time_fit_rms_l2_relative_error = 1.058e-1
frequency_vs_time_series_rms_l2_relative_error = 1.031e-1
motion_spectrum_density_l2_relative_error = 1.971e-1

frequency_rms_max = 0.274432
time_fit_rms_max = 0.273284
time_series_rms_max = 0.272302

frequency_rms_mean = 0.176632
time_fit_rms_mean = 0.193194
time_series_rms_mean = 0.192561
```

The representative motion spectra match well around the dominant wave peak.
The main RMS mismatch comes from the stern-side centerline region near
`x/L = 1`, where the time-domain RMS is about `0.029 m` above the direct
frequency-domain spectrum-integral RMS.

## Original 42-Frequency Case

Command:

```powershell
.\.venv\Scripts\python.exe scripts\compare_spectrum_frequency_time_response.py --case-root results\time_domain\hydrodynamic_extrapolation_dm10_mesh2\time_domain_original --output-root results\time_domain\spectrum_frequency_time_comparison_original
```

Metrics:

```text
component_count = 42
frequency_vs_time_fit_rms_l2_relative_error = 8.958e-2
frequency_vs_time_series_rms_l2_relative_error = 7.479e-2
motion_spectrum_density_l2_relative_error = 1.667e-1

frequency_rms_max = 0.274634
time_fit_rms_max = 0.271352
time_series_rms_max = 0.268380
```

## Interpretation

The frequency-domain and time-domain spectrum results are in the same
magnitude range and agree closely near the dominant spectral peak. The current
remaining RMS mismatch is roughly `7%` to `11%` depending on whether the
original or extrapolated frequency grid is used.

This difference is expected at the current development stage because the
time-domain Cummins solver uses a finite-band radiation-memory approximation,
whereas the frequency-domain reference uses the direct `A(omega), B(omega)`
matrices at every wave component. A state-space radiation approximation or a
multi-frequency residual strategy should be evaluated next if the goal is to
make spectrum-integrated RMS match the frequency-domain RAO integral more
tightly across the full flexible body.

## Spectrum-Focused Mesh2 Dense Grid Update

The first spectrum-comparison plots were not sufficiently convincing because
the JONSWAP peak was under-resolved. A new spectrum-focused mesh2 BEM dataset
was generated with:

```text
spectrum = JONSWAP
Hs = 1.0 m
Tp = 15.11471086644115 s
gamma = 3.3
target omega = 0.4157 rad/s
omega range = 0.1 to 2.0 rad/s
omega count = 88
points in half-power band = 11
points in 5-95 percent energy band = 42
hydrodynamic dataset = data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
dataset SHA256 = A1D40E3C365C20B641FF14646AB07D61A2518C5529F49CD569E8949C62D5C70F
generation elapsed = 961.190 s
```

The dense grid design artifacts are:

```text
results/time_domain/spectrum_frequency_grid_design_omega0p10/omega_values.txt
results/time_domain/spectrum_frequency_grid_design_omega0p10/spectrum_frequency_grid_metrics.json
results/time_domain/spectrum_frequency_grid_design_omega0p10/figures/spectrum_focused_frequency_grid.png
```

Dense-grid Cummins validation without residual correction:

```text
original 88-frequency case:
  frequency_vs_time_fit_rms_l2_relative_error = 9.901e-2
  frequency_vs_time_series_rms_l2_relative_error = 5.907e-2
  motion_spectrum_density_l2_relative_error = 1.917e-1

extrapolated 188-frequency case:
  frequency_vs_time_fit_rms_l2_relative_error = 1.163e-1
  frequency_vs_time_series_rms_l2_relative_error = 7.626e-2
  motion_spectrum_density_l2_relative_error = 2.309e-1
```

These results show that the grid is now adequate for the input spectrum, but
the finite-band Cummins radiation kernel still does not exactly reproduce the
frequency-domain `A(omega), B(omega)` operator across the response band.

## Selected-Frequency Residual Correction

The adapter already supports a WEC-Sim-like selected-frequency residual:

```text
radiation_residual_model = selected_frequency
```

This correction preserves the external adapter boundary and does not modify
the RODM frequency-domain core. It closes the discrete Cummins radiation
operator against the original frequency-domain added mass and radiation
damping at the spectrum peak frequency.

With the same dense mesh2 spectrum case:

```text
original 88-frequency case, selected-frequency residual:
  frequency_vs_time_fit_rms_l2_relative_error = 3.301e-3
  frequency_vs_time_series_rms_l2_relative_error = 4.343e-2
  motion_spectrum_density_l2_relative_error = 6.896e-3

extrapolated 188-frequency case, selected-frequency residual:
  frequency_vs_time_fit_rms_l2_relative_error = 3.415e-3
  frequency_vs_time_series_rms_l2_relative_error = 4.348e-2
  motion_spectrum_density_l2_relative_error = 7.172e-3
```

The harmonic-fit spectrum now overlays the frequency-domain spectrum. The
remaining `4.3%` time-series RMS difference mainly comes from finite-length
random-wave realization, ramp/discard treatment, and direct RMS estimation,
not from a frequency-domain/time-domain transfer-function mismatch.

The reproducible outputs are:

```text
results/time_domain/hdro_dense88_res_peak/hydrodynamic_extrapolation_metrics.json
results/time_domain/spec_cmp_dense88_res_peak_original/frequency_time_motion_spectrum_metrics.json
results/time_domain/spec_cmp_dense88_res_peak_original/figures/wave_and_midpoint_motion_spectrum.png
results/time_domain/spec_cmp_dense88_res_peak_original/figures/centerline_rms_frequency_vs_time.png
results/time_domain/spec_cmp_dense88_res_peak_extrapolated/frequency_time_motion_spectrum_metrics.json
```

For the residual-corrected hydrodynamic extrapolation validation:

```text
added_mass_original_range_delta = 0
radiation_damping_original_range_delta = 0
kernel_tail_rms_after_over_before = 0.602355
centerline_heave_rms_closure_error before/after = 4.827e-2 / 4.871e-2
```

Current conclusion: for a spectrum centered near `0.4157 rad/s`, the validated
baseline should use the dense spectrum-focused grid plus
`radiation_residual_model=selected_frequency`. A future state-space radiation
model or multi-frequency residual strategy is still needed for broader
sea-state spectra where energy is not concentrated near one dominant peak.

## Multi-Seed Statistical Check

A five-seed random-phase sweep was added:

```text
script = scripts/validate_spectrum_seed_sweep.py
output = results/time_domain/seed_sweep_dense88_res_peak/
seeds = 20260522, 20260523, 20260524, 20260525, 20260526
radiation_residual_model = selected_frequency
```

Summary:

```text
frequency_vs_time_fit_rms_l2_relative_error:
  mean = 3.290e-3
  std  = 8.951e-6
  min  = 3.276e-3
  max  = 3.301e-3

motion_spectrum_density_l2_relative_error:
  mean = 6.849e-3
  std  = 2.892e-5
  min  = 6.812e-3
  max  = 6.896e-3

frequency_vs_time_series_rms_l2_relative_error:
  mean = 3.202e-2
  std  = 2.859e-2
  min  = 4.920e-3
  max  = 8.255e-2
```

The harmonic-fit transfer-function error is seed-insensitive and remains near
`0.33%`. The direct time-series RMS has larger seed-to-seed scatter, which is
consistent with finite-duration random-wave RMS estimation.

## Radiation Reconstruction Diagnostic

The Cummins radiation kernel was also transformed back to frequency-domain
coefficients:

```text
script = scripts/diagnose_radiation_reconstruction.py
baseline output = results/time_domain/rad_recon_dense88_none/
residual output = results/time_domain/rad_recon_dense88_res_peak/
```

Without residual correction:

```text
weighted added-mass reconstruction error = 4.019e-2
weighted radiation-damping reconstruction error = 2.567e-2
selected-omega added-mass error = 3.802e-2
selected-omega radiation-damping error = 4.719e-2
```

With `selected_frequency` residual correction:

```text
weighted added-mass reconstruction error = 5.659e-3
weighted radiation-damping reconstruction error = 3.123e-2
selected-omega added-mass error = 5.323e-20
selected-omega radiation-damping error = 1.024e-19
```

Over the 5%-95% wave-energy band:

```text
added-mass error: 3.927e-2 -> 3.659e-3
radiation-damping error: 1.948e-2 -> 2.903e-2
```

Interpretation: the selected-frequency residual exactly closes the discrete
Cummins radiation operator at the spectrum peak and greatly improves the
added-mass error over the active wave band. It is not a broadband optimal
radiation model, so a later state-space or multi-frequency residual model is
still the right next step for wide-band or multi-peak seas.

## Numerical Sensitivity Update

A sensitivity matrix was added for time-step density and radiation-memory
duration:

```text
script = scripts/validate_cummins_numerical_sensitivity.py
coarse matrix = results/time_domain/num_sens_dense88_res_peak/
memory focus = results/time_domain/num_sens_dense88_memory_focus/
step focus = results/time_domain/num_sens_dense88_step_focus/
```

Coarse matrix:

```text
memory_cycles = 2, 4, 6
steps_per_peak_cycle = 30, 40, 60

fit RMS error min/max = 1.509e-3 / 1.054e-2
motion spectrum error min/max = 4.184e-3 / 2.108e-2
```

Best point in the coarse matrix:

```text
memory_cycles = 2
steps_per_peak_cycle = 60
frequency_vs_time_fit_rms_l2_relative_error = 1.509e-3
motion_spectrum_density_l2_relative_error = 4.184e-3
```

Focused checks:

```text
memory sweep at 60 steps/cycle:
  best = 2 memory cycles, fit RMS error = 1.509e-3

time-step sweep at 2 memory cycles:
  40 steps/cycle -> 2.265e-3
  60 steps/cycle -> 1.509e-3
  80 steps/cycle -> 1.426e-3
```

Current recommended default for the validated JONSWAP-DM10 mesh2 baseline:

```text
memory_cycles = 2
steps_per_peak_cycle = 60
radiation_convolution_rule = trapezoidal
radiation_passivity_correction = clip_negative_eigenvalues
radiation_residual_model = selected_frequency
```

This setting keeps harmonic-fit frequency/time RMS error near `0.15%` while
avoiding the extra cost of 80 steps/cycle. Longer memory windows are not
monotonically better because the finite-band Cummins kernel still contains
tail oscillation error.
