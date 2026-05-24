# Time-Domain Implementation Log - 2026-05-21

This note records the current long-run implementation state for the RODM
time-domain solver.

## Completed in this step

- Added `radiation_model` to `TimeDomainSimulationConfig`.
- Supported two RODM time-domain radiation models:
  - `constant`: selected-frequency bridge using `A(omega_ref)` and
    `B(omega_ref)`.
  - `direct_convolution`: Cummins-style radiation memory using `A_inf` and
    an IRF built from the BEM radiation-damping frequency grid.
- Added `src/offshore_energy_sim/time_domain/rodm_hydrodynamics.py`.
- Added RODM preprocessing for:
  - multi-frequency `added_mass`;
  - multi-frequency `radiation_damping`;
  - hydrostatic stiffness;
  - selected-frequency wave excitation;
  - hydrodynamic DOF removal;
  - optional hydrodynamic node-block reversal;
  - omega-grid sorting and validation;
  - high-frequency-tail estimate of `A_inf`.
- Connected `direct_convolution` into `solve_rodm_time_domain_case`.
- Saved optional direct-convolution outputs from runner scripts:
  - `memory_force_time.npy`;
  - `radiation_irf_time.npy`;
  - `radiation_irf.npy`;
  - `added_mass_infinite.npy`.
- Added diagnostic figures in the 300 m reference time-series script:
  - `radiation_memory_force_norm.png`;
  - `radiation_irf_norm.png`.
- Added CLI controls to:
  - `scripts/run_rodm_case_from_config.py`;
  - `scripts/run_time_domain_reference_case_300.py`.
- Updated `configs/templates/rodm_frequency_case.yaml` with the time-domain
  radiation-model settings.
- Added unit tests for config validation, zero IRF consistency, `A_inf`
  estimation, and synthetic xarray hydrodynamic preprocessing.

## Current command examples

Single-frequency validation bridge:

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py `
  --data-root D:\RODM-data\DM-FEM2D `
  --radiation-model constant
```

Cummins direct-convolution path:

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py `
  --data-root D:\RODM-data\DM-FEM2D `
  --radiation-model direct_convolution `
  --memory-duration 60
```

Generic config runner:

```powershell
.\.venv\Scripts\python.exe scripts\run_rodm_case_from_config.py `
  --config configs\reference_case_300.yaml `
  --domain time `
  --radiation-model direct_convolution `
  --memory-duration 60
```

## Verification run

Local unit and integration-light tests:

```text
43 passed
```

The real 300 m RODM case still requires these external files:

```text
C:\Users\WYJ\data\DM-FEM2D\HydrodynamicData\Yoga\DM10_300_direction0.nc
C:\Users\WYJ\data\DM-FEM2D\StructureData\JobMesh5_5_MASS1.mtx
C:\Users\WYJ\data\DM-FEM2D\StructureData\JobMesh5_5_STIF1.mtx
```

Without them, the reference-case script exits in `missing_inputs` mode and
writes `results/time_domain/reference_case_300_timeseries/metrics.json`.

## Local data mirror

The 300 m single-frequency benchmark inputs have been copied into the current
working tree under:

```text
data/external/DM-FEM2D/
```

See `local_data_layout_300m.md` for the exact file list and hashes. The
time-domain reference scripts now prefer this repo-local data mirror when
`--data-root` and `RODM_DM_FEM_ROOT` are not provided.

Important limitation: `DM10_300_direction0.nc` contains one omega value. It is
valid for `radiation_model=constant`, but `radiation_model=direct_convolution`
still requires a compatible multi-frequency hydrodynamic dataset.

## Basic 300 m Time-Series Result

The basic 300 m, 10-module time-domain case has been run with:

```text
radiation_model = constant
structural_reduction_method = serep_ridge
cycles = 80
steps_per_cycle = 180
ramp_cycles = 5
discard_cycles = 55
```

The legacy square `serep` reduction is not suitable as a long-time time-domain
default because its reduced mass/stiffness matrices are not positive definite.
`serep_ridge` keeps the same SEREP family but regularizes the master modal
mapping enough for stable time integration.

Completed output:

```text
results/time_domain/reference_case_300_timeseries/
```

Key metrics:

```text
time_samples = 14401
global_displacement_shape = (14401, 3965)
centerline_heave_shape = (14401, 60)
global_l2_relative_error = 4.819467436e-4
master_l2_relative_error = 4.884345187e-4
```

Main figures:

```text
figures/centerline_representative_heave_time.png
figures/centerline_representative_heave_final_5_cycles.png
figures/centerline_heave_snapshots.png
figures/centerline_heave_frequency_validation.png
```

## Cummins Direct-Convolution Validation

The WEC-Sim-style Cummins equation path is now implemented as:

```text
(M_struct + A_inf) qdd + K q + integral_0^t K_rad(t-tau) qd(tau) dtau = F_exc(t)
```

Implementation details:

- IRF from radiation damping:
  `K_rad(t) = 2/pi * integral B(omega) cos(omega t) d omega`.
- Frequency-domain recovery diagnostics from IRF:
  `B(omega) = integral K_rad(t) cos(omega t) dt`.
- Added-mass consistency diagnostic:
  `A(omega) = A_inf - 1/omega * integral K_rad(t) sin(omega t) dt`.
- The zero-lag IRF term is treated implicitly inside the Newmark effective
  damping matrix.
- A passivity correction option can symmetrize and clip negative eigenvalues
  of radiation damping and `A_inf` before time integration.

Validation script:

```text
scripts/run_cummins_time_domain_validation.py
```

Real BM10 multi-frequency result:

```text
results/time_domain/cummins_bm10_validation/
```

Key metrics from the current validation run:

```text
hydrodynamic dataset = BM10_direaction0_full.nc
target omega = 0.4157 rad/s
selected omega = 0.3923076923 rad/s
frequency index = 6
radiation_passivity_correction = clip_negative_eigenvalues
constant_global_l2_relative_error = 2.519e-3
cummins_global_l2_relative_error = 2.315e-1
irf_damping_relative_error_at_selected_omega = 2.374e-1
irf_added_mass_relative_error_at_selected_omega = 6.601e-1
```

The direct-convolution equation is stable and produces comparison plots. The
remaining error is dominated by hydrodynamic reconstruction: the available
multi-frequency BM10 file does not contain the exact 300 m benchmark omega, and
the finite-band IRF/A_inf reconstruction does not yet reproduce the selected
frequency-domain `A(omega), B(omega)` closely enough for final validation.

## Locally Generated DM10 Multi-Frequency Dataset

Added a reproducible Capytaine entry point:

```text
scripts/generate_dm10_cummins_hydrodynamics.py
```

Generated datasets:

```text
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_9plus_target_mesh4.nc
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh4.nc
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc
```

The 42-frequency dataset contains the exact target omega `0.4157 rad/s`.
It uses the 10-module Yoga geometry, `rho=1025 kg/m3`, water depth `58.5 m`,
and applies the historical Yoga static convention by scaling the hydrostatic
matrix to `rho_static=1000 kg/m3`.

Key generation result:

```text
omega_count = 42
problem_count = 2562
mesh_size = 4 m
elapsed_seconds = 31.379
sha256 = D685585645470B420194A6BD77F78AAF31E614D585B4EADB0857AA9EEAC3FB56
```

Validation result with the discrete selected-frequency residual:

```text
results/time_domain/cummins_dm10_generated_mesh4_42freq_residual_discrete_validation/
constant_global_l2_relative_error = 3.915e-3
cummins_global_l2_relative_error = 5.003e-3
corrected_damping_relative_error_at_selected_omega = 1.788e-16
corrected_added_mass_relative_error_at_selected_omega = 1.141e-17
```

Important interpretation: the raw finite-band IRF still has large continuous
coefficient-recovery error, so the residual correction is a regular-wave
validation bridge, not the final broadband radiation model. It closes the
discrete Cummins solver at the selected frequency and lets the time-domain
RODM workflow produce validated time histories while the higher-quality
IRF/state-space work continues.

## Mesh-2 Cummins Research Step

Generated a finer 42-frequency DM10 dataset:

```text
dataset = DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc
omega_count = 42
problem_count = 2562
mesh_size = 2 m
elapsed_seconds = 274.654
sha256 = C6AF9671D59198CC15D65CBE91160DF4892D432D6EE3AAD25FF17D1F34DCA002
```

Target-frequency mesh sensitivity relative to the mesh-4 dataset:

```text
added_mass_relative_difference = 1.997e-2
radiation_damping_relative_difference = 2.523e-2
excitation_force_relative_difference = 1.059e-2
```

Added research controls:

```text
radiation_frequency_window = none | linear_tail | cosine_tail
radiation_window_start_omega = optional taper start
radiation_window_stop_omega = optional taper stop
radiation_convolution_rule = rectangular | trapezoidal
```

New sensitivity script:

```text
scripts/analyze_cummins_irf_sensitivity.py
```

Mesh-2 IRF sensitivity outputs:

```text
results/time_domain/cummins_irf_sensitivity_dm10_mesh2/
results/time_domain/cummins_irf_sensitivity_dm10_mesh2_trapezoidal/
```

Representative time-domain validation metrics:

```text
mesh2, no window, no residual, rectangular:
  cummins_global_l2_relative_error = 1.693e-1
  irf_damping_relative_error = 4.797e-2
  irf_added_mass_relative_error = 3.957e-2

mesh2, no window, selected-frequency residual, rectangular:
  cummins_global_l2_relative_error = 3.651e-3

mesh2, no window, no residual, trapezoidal:
  cummins_global_l2_relative_error = 1.286e-1

mesh2, no window, selected-frequency residual, trapezoidal:
  cummins_global_l2_relative_error = 3.652e-3

mesh2, linear_tail from 1.0 rad/s, no residual, trapezoidal:
  cummins_global_l2_relative_error = 1.122e-1
  irf_damping_relative_error = 2.668e-2
  irf_added_mass_relative_error = 7.610e-2

mesh2, linear_tail from 1.0 rad/s, selected-frequency residual, trapezoidal:
  cummins_global_l2_relative_error = 3.696e-3
```

Current conclusion: mesh refinement greatly improves the raw continuous
Cummins reconstruction. Windowing can reduce the damping reconstruction error,
but it can worsen added-mass consistency. The most reliable regular-wave
setting for the current RODM validation is therefore:

```text
hydrodynamic dataset = mesh2 42-frequency DM10
radiation_frequency_window = none
radiation_convolution_rule = trapezoidal or rectangular
radiation_residual_model = selected_frequency
```

The residual-free broadband model still needs either better finite-band
extrapolation or a state-space radiation fit before irregular-wave production
use.

## Spectrum and External-Force Excitation

Added WEC-Sim-like time-domain excitation inputs:

```text
excitation_model = regular_wave | wave_spectrum | external_force
spectrum_type = jonswap | pierson_moskowitz
external_force = user-supplied reduced-DOF force time series
```

Core additions:

```text
src/offshore_energy_sim/time_domain/excitation.py
  jonswap_spectrum
  pierson_moskowitz_spectrum
  spectral_wave_amplitudes
  spectral_wave_force_time_series
  wave_elevation_time_series
  external_force_time_series

src/offshore_energy_sim/time_domain/solver.py
  solve_rodm_time_domain_case now builds force histories from
  regular waves, wave spectra, or external-force arrays.
```

User-facing script:

```text
scripts/run_time_domain_excitation_case.py
```

JONSWAP demo:

```text
results/time_domain/spectrum_jonswap_dm10_mesh2_demo/
hydrodynamic dataset = mesh2 42-frequency DM10
H_s = 1.0 m
peak_period = 15.114710866 s
time_samples = 721
centerline_heave_rms_max = 2.893e-1
centerline_heave_abs_max = 6.843e-1
wave_elevation_rms = 2.554e-1
wave_elevation_abs_max = 5.933e-1
```

External-force demo:

```text
results/time_domain/external_force_dm10_mesh2_demo/
input CSV = results/time_domain/external_force_demo_input.csv
external force = 1e6 N cosine on one retained heave-like DOF
time_samples = 640
centerline_heave_rms_max = 4.881e-2
centerline_heave_abs_max = 7.687e-2
```

## Spectrum Statistical Validation

Added a post-processing validation script for irregular-wave time-domain
statistics:

```text
scripts/validate_spectrum_time_domain_statistics.py
```

The script checks the chain:

```text
target wave spectrum
  -> discrete wave components
  -> synthesized wave-elevation time series
  -> fitted wave/force/response harmonic components
  -> centerline response RMS
```

New post-processing helpers:

```text
src/offshore_energy_sim/time_domain/postprocess.py
  fit_multi_harmonic_amplitudes
  harmonic_component_variance
  zero_mean_rms
  relative_l2_error
```

Short JONSWAP demo validation:

```text
case = results/time_domain/spectrum_jonswap_dm10_mesh2_demo/
time_samples = 721
discard = 2 peak periods
wave_component_amplitude_l2_relative_error = 3.0e-15
wave_variance_time_vs_component_relative_error = 1.908e-1
centerline_heave_rms_l2_relative_error = 8.364e-2
```

The short case exactly recovers the deterministic component amplitudes, but the
plain time-series variance is biased by the short finite realization.

Long JONSWAP statistics validation:

```text
case = results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/
time_samples = 3201
duration = 1209.176869 s
discard = 5 peak periods
target_trapz_wave_variance = 6.250000e-2
discrete_component_wave_variance = 6.250439e-2
fitted_component_wave_variance = 6.250439e-2
time_wave_variance = 6.126020e-2
discrete_component_Hs = 1.000035
time_series_Hs = 0.990032
wave_component_amplitude_l2_relative_error = 3.043e-15
wave_component_complex_l2_relative_error = 1.079e-14
wave_variance_time_vs_component_relative_error = 1.991e-2
wave_variance_fit_vs_component_relative_error = 4.441e-16
centerline_heave_rms_l2_relative_error = 1.594e-2
excitation_force_rms_l2_relative_error = 1.144e-2
```

Validation products:

```text
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/statistics_metrics.json
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/centerline_rms_validation.csv
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/figures/wave_variance_closure.png
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/figures/wave_component_recovery.png
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/figures/centerline_heave_rms_closure.png
results/time_domain/spectrum_jonswap_dm10_mesh2_long_stats/statistics_validation/figures/representative_heave_component_spectra.png
```

Current conclusion: the spectrum-driven time-domain path is internally
consistent. Component fitting recovers amplitude and phase to roundoff, while
time-series RMS convergence improves as the realization length increases. The
remaining response-RMS differences are therefore dominated by finite-sample
statistics and startup-discard choices, not by a broken wave-force or phase
implementation.

## Adapter-Layer Hydrodynamic Extrapolation

Added an external `time_domain_adapter` package for WEC-Sim/Cummins-style
preprocessing without modifying the RODM frequency-domain core:

```text
src/offshore_energy_sim/time_domain_adapter/
  hydrodynamic_extrapolation.py
  radiation_kernel.py
  cummins_solver.py
  wecsim_like_validation.py
  extrapolation_diagnostics.py

scripts/validate_hydrodynamic_extrapolation.py
tests/test_hydrodynamic_extrapolation.py
tests/test_radiation_kernel.py
```

The adapter reads an exported hydrodynamic NetCDF file and writes a separate
extrapolated copy:

```text
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/hydrodynamics/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2_adapter_extrapolated.nc
```

The original DM10 mesh2 grid is embedded unchanged inside the extended grid:

```text
original omega range = 0.10 to 2.00 rad/s, 42 frequencies
extended omega range = 0.02 to 8.00 rad/s, 142 frequencies
```

Original-range invariance:

```text
max_abs_difference_inside_original_range:
  added_mass = 0.0
  radiation_damping = 0.0
  wave_force = 0.0
  Froude_Krylov_force = 0.0
  diffraction_force = 0.0
  excitation_force = 0.0
```

Radiation-kernel comparison:

```text
tail_rms_to_peak_ratio before = 3.694e-3
tail_rms_to_peak_ratio after  = 2.853e-3
after / before                = 0.772

tail_peak_to_peak_ratio before = 7.357e-3
tail_peak_to_peak_ratio after  = 5.638e-3
after / before                 = 0.766

norm_oscillation_score before = 0.472
norm_oscillation_score after  = 0.358
```

Long JONSWAP time-domain comparison:

```text
wave_variance_closure_error:
  before = 1.991e-2
  after  = 6.708e-3

excitation_force_rms_closure_error:
  before = 1.144e-2
  after  = 4.007e-3

centerline_heave_rms_closure_error:
  before = 1.594e-2
  after  = 3.358e-3

centerline_heave_rms_max:
  before = 0.270173
  after  = 0.268083

radiation_force_rms:
  before = 6.074e6
  after  = 6.210e6
```

Diagnostic report:

```text
docs/time_domain/time_domain_adapter_extrapolation_report_2026_05_22.md
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/hydrodynamic_extrapolation_metrics.json
```

Current conclusion: the adapter-layer extrapolation is opt-in and preserves
the RODM frequency-domain data. For the current DM10 mesh2 validation case it
improves normalized radiation-kernel tail metrics and improves the long
JONSWAP response RMS closure.

## Basic Benchmark Validation Gate

Added a unified validation script:

```text
scripts/validate_time_domain_basic_benchmark.py
```

It builds a benchmark dashboard from:

```text
1. SDOF Newmark/frequency-domain validation.
2. RODM regular-wave constant A/B validation.
3. RODM Cummins direct-convolution validation.
4. Corrected radiation-coefficient reconstruction.
5. JONSWAP-DM10 mesh2 spectrum statistics.
6. Adapter extrapolation and radiation-kernel stability.
```

Current benchmark result:

```text
status = passed
SDOF Newmark relative error                 = 3.883e-5
constant A/B time-domain vs frequency error = 4.008e-3
Cummins time-domain vs frequency error      = 3.652e-3
corrected B reconstruction error            = 6.924e-19
original wave variance closure error        = 1.991e-2
extrapolated wave variance closure error    = 6.708e-3
original heave RMS closure error            = 1.594e-2
extrapolated heave RMS closure error        = 3.358e-3
kernel tail RMS after/before                = 7.722e-1
```

Outputs:

```text
results/time_domain/basic_benchmark_validation/basic_benchmark_metrics.json
results/time_domain/basic_benchmark_validation/figures/basic_benchmark_dashboard.png
results/time_domain/basic_benchmark_validation/figures/validation_error_summary.png
results/time_domain/basic_benchmark_validation/figures/spectrum_closure_before_after.png
docs/time_domain/basic_benchmark_validation_2026_05_22.md
```

This benchmark is the new validation gate before state-space radiation
approximation or other WEC-Sim-like extensions.

## Spectrum Frequency-Time Response Comparison

Added a direct comparison between frequency-domain spectrum integration and
time-domain random-wave response fitting:

```text
scripts/compare_spectrum_frequency_time_response.py
```

The script computes:

```text
frequency-domain:
  S_z(omega) = |RAO_z(omega)|^2 S_eta(omega)
  RMS_z = sqrt(integral S_z(omega) d omega)

time-domain:
  fit z(t) = sum_j Re(Z_j exp(-i omega_j t))
  S_z,time(omega_j) = 0.5 |Z_j|^2 / Delta omega_j
  RMS_z = sqrt(0.5 sum_j |Z_j|^2)
```

Extrapolated 142-frequency JONSWAP case:

```text
case = results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/time_domain_extrapolated
frequency_vs_time_fit_rms_l2_relative_error = 1.058e-1
frequency_vs_time_series_rms_l2_relative_error = 1.031e-1
motion_spectrum_density_l2_relative_error = 1.971e-1
frequency_rms_max = 0.274432
time_fit_rms_max = 0.273284
time_series_rms_max = 0.272302
```

Original 42-frequency JONSWAP case:

```text
case = results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/time_domain_original
frequency_vs_time_fit_rms_l2_relative_error = 8.958e-2
frequency_vs_time_series_rms_l2_relative_error = 7.479e-2
motion_spectrum_density_l2_relative_error = 1.667e-1
frequency_rms_max = 0.274634
time_fit_rms_max = 0.271352
time_series_rms_max = 0.268380
```

Outputs:

```text
results/time_domain/spectrum_frequency_time_comparison/frequency_time_motion_spectrum_metrics.json
results/time_domain/spectrum_frequency_time_comparison/figures/motion_spectrum_frequency_vs_time.png
results/time_domain/spectrum_frequency_time_comparison/figures/centerline_rms_frequency_vs_time.png
results/time_domain/spectrum_frequency_time_comparison/figures/wave_and_midpoint_motion_spectrum.png
docs/time_domain/spectrum_frequency_time_response_comparison_2026_05_22.md
```

Current conclusion: the time-domain random-wave response spectrum agrees with
the direct frequency-domain spectrum near the dominant wave peak, but the
centerline RMS differs by roughly 7-11%. The largest mismatch is near the
stern-side centerline region. This is now a concrete target for state-space
radiation approximation or a multi-frequency residual correction.

## Dense Spectrum Grid and Residual-Corrected Cummins Closure

The earlier spectrum comparison had too few frequency points around the
JONSWAP peak. A spectrum-focused grid was therefore designed and used to
generate a new mesh2 hydrodynamic dataset:

```text
script = scripts/design_spectrum_frequency_grid.py
grid output = results/time_domain/spectrum_frequency_grid_design_omega0p10/omega_values.txt
omega range = 0.1 to 2.0 rad/s
omega count = 88
points in half-power band = 11
points in 5-95 percent energy band = 42
hydrodynamic dataset = data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
SHA256 = A1D40E3C365C20B641FF14646AB07D61A2518C5529F49CD569E8949C62D5C70F
Capytaine elapsed = 961.190 s
```

The time-domain validation now supports the existing selected-frequency
radiation residual from the command line:

```text
scripts/validate_hydrodynamic_extrapolation.py --radiation-residual-model selected_frequency
```

This keeps the architecture unchanged: RODM remains the independent
frequency-domain core, and the residual correction is applied only by the
external time-domain adapter.

Dense-grid results without residual correction:

```text
original 88-frequency:
  frequency_vs_time_fit_rms_l2_relative_error = 9.901e-2
  frequency_vs_time_series_rms_l2_relative_error = 5.907e-2
  motion_spectrum_density_l2_relative_error = 1.917e-1

extrapolated 188-frequency:
  frequency_vs_time_fit_rms_l2_relative_error = 1.163e-1
  frequency_vs_time_series_rms_l2_relative_error = 7.626e-2
  motion_spectrum_density_l2_relative_error = 2.309e-1
```

Dense-grid results with selected-frequency residual correction:

```text
original 88-frequency:
  frequency_vs_time_fit_rms_l2_relative_error = 3.301e-3
  frequency_vs_time_series_rms_l2_relative_error = 4.343e-2
  motion_spectrum_density_l2_relative_error = 6.896e-3

extrapolated 188-frequency:
  frequency_vs_time_fit_rms_l2_relative_error = 3.415e-3
  frequency_vs_time_series_rms_l2_relative_error = 4.348e-2
  motion_spectrum_density_l2_relative_error = 7.172e-3
```

Hydrodynamic extrapolation with the residual-corrected time-domain comparison:

```text
results/time_domain/hdro_dense88_res_peak/hydrodynamic_extrapolation_metrics.json
added_mass_original_range_delta = 0
radiation_damping_original_range_delta = 0
kernel_tail_rms_after_over_before = 6.024e-1
centerline_heave_rms_closure_error before/after = 4.827e-2 / 4.871e-2
```

Current conclusion: the unusual earlier spectrum plot was caused by two
separate issues. First, the wave spectrum peak needed a denser local omega
grid. Second, a finite-band Cummins kernel is not identical to the direct
frequency-domain `A(omega), B(omega)` operator unless a residual or a more
accurate radiation approximation is used. For the current JONSWAP-DM10 mesh2
baseline, the validated setting is:

```text
omega grid = spectrum-focused dense 88-point grid
radiation_convolution_rule = trapezoidal
radiation_passivity_correction = clip_negative_eigenvalues
radiation_residual_model = selected_frequency
```

## Stepwise Validation After Dense-Grid Baseline

Two additional validation scripts were added:

```text
scripts/validate_spectrum_seed_sweep.py
scripts/diagnose_radiation_reconstruction.py
```

The seed sweep runs repeated random-phase JONSWAP simulations while reusing the
same frequency-domain RAO reference. Five seeds were run:

```text
output = results/time_domain/seed_sweep_dense88_res_peak/
seed_count = 5
fit RMS error mean = 3.290e-3
fit RMS error std  = 8.951e-6
motion spectrum error mean = 6.849e-3
motion spectrum error std  = 2.892e-5
time-series RMS error mean = 3.202e-2
time-series RMS error std  = 2.859e-2
```

Conclusion: the harmonic-fit response spectrum is nearly seed-invariant, while
direct time-series RMS still has finite-duration random-wave scatter.

The radiation reconstruction diagnostic transforms the Cummins kernel back to
frequency-domain hydrodynamic coefficients and compares them with the original
RODM/BEM `A(omega), B(omega)` data:

```text
without residual:
  weighted A reconstruction error = 4.019e-2
  weighted B reconstruction error = 2.567e-2
  selected-omega A error = 3.802e-2
  selected-omega B error = 4.719e-2

with selected_frequency residual:
  weighted A reconstruction error = 5.659e-3
  weighted B reconstruction error = 3.123e-2
  selected-omega A error = 5.323e-20
  selected-omega B error = 1.024e-19
```

Over the 5%-95% wave-energy band:

```text
A reconstruction error = 3.927e-2 -> 3.659e-3
B reconstruction error = 1.948e-2 -> 2.903e-2
```

This confirms why the residual-corrected time-domain motion spectrum closes so
well for the current narrow JONSWAP case: the selected-frequency residual
exactly closes the discrete Cummins radiation operator at the dominant wave
frequency. It also confirms the limitation: selected-frequency residual is a
single-peak validation tool, not yet a broadband radiation model.

## Cummins Numerical Sensitivity Matrix

Added a numerical sensitivity sweep for the residual-corrected dense-grid
JONSWAP baseline:

```text
script = scripts/validate_cummins_numerical_sensitivity.py
coarse output = results/time_domain/num_sens_dense88_res_peak/
memory focus output = results/time_domain/num_sens_dense88_memory_focus/
step focus output = results/time_domain/num_sens_dense88_step_focus/
```

The coarse matrix used:

```text
memory_cycles = 2, 4, 6
steps_per_peak_cycle = 30, 40, 60
radiation_residual_model = selected_frequency
```

Coarse-matrix summary:

```text
frequency_vs_time_fit_rms_l2_relative_error:
  min = 1.509e-3
  max = 1.054e-2

motion_spectrum_density_l2_relative_error:
  min = 4.184e-3
  max = 2.108e-2
```

The best coarse-grid point was:

```text
memory_cycles = 2
steps_per_peak_cycle = 60
frequency_vs_time_fit_rms_l2_relative_error = 1.509e-3
motion_spectrum_density_l2_relative_error = 4.184e-3
```

Focused memory sweep at 60 steps/cycle:

```text
memory 1.0 cycles: fit RMS error = 4.476e-3
memory 1.5 cycles: fit RMS error = 1.600e-3
memory 2.0 cycles: fit RMS error = 1.509e-3
memory 2.5 cycles: fit RMS error = 2.268e-3
memory 3.0 cycles: fit RMS error = 1.744e-3
memory 4.0 cycles: fit RMS error = 2.233e-3
```

Focused time-step sweep at 2 memory cycles:

```text
40 steps/cycle: fit RMS error = 2.265e-3
60 steps/cycle: fit RMS error = 1.509e-3
80 steps/cycle: fit RMS error = 1.426e-3
```

Current numerical recommendation for this JONSWAP-DM10 mesh2 baseline:

```text
memory_cycles = 2
steps_per_peak_cycle = 60
radiation_convolution_rule = trapezoidal
radiation_passivity_correction = clip_negative_eigenvalues
radiation_residual_model = selected_frequency
```

`80 steps/cycle` gives a small additional improvement, but `60 steps/cycle`
is a better default compromise for longer random-wave simulations. Longer
memory durations are not automatically better here because the finite-band
Cummins kernel has residual tail oscillations; overly long memory can
reintroduce those tail errors into the direct convolution.

## State-Space Radiation Approximation Prototype

Added the first external state-space radiation-memory prototype:

```text
src/offshore_energy_sim/time_domain_adapter/state_space_radiation.py
src/offshore_energy_sim/time_domain_adapter/state_space_solver.py
src/offshore_energy_sim/time_domain_adapter/mooring.py
scripts/validate_state_space_radiation.py
scripts/validate_state_space_order_sweep.py
scripts/validate_state_space_response.py
tests/test_state_space_radiation.py
tests/test_mooring_adapter.py
docs/time_domain/state_space_radiation_validation_2026_05_22.md
docs/time_domain/state_space_mooring_validation_2026_05_22.md
```

This stays inside the adapter layer. The RODM frequency-domain core does not
depend on the state-space code.

Two fitting routes were tested:

```text
common real-pole exponential fit:
  K(t) ~= sum_i R_i exp(-p_i t)

ERA discrete state-space fit:
  G_k = dt K(k dt) ~= C A^(k-1) B
```

The common real-pole fit is useful as a simple baseline, but it is not accurate
enough for the current dense-grid DM10 radiation kernel:

```text
order 8:
  kernel L2 error = 2.865e-1
  state/direct memory-force error = 2.291e-1

order 12 with ridge_alpha = 1e-6:
  kernel L2 error = 2.941e-1
  state/direct memory-force error = 2.060e-1
```

The ERA route is the current state-space candidate:

```text
ERA order 120, block 40x40:
  kernel L2 error = 1.911e-2
  state/direct A weighted error = 4.847e-3
  state/direct B weighted error = 2.283e-2
  state/direct memory-force error = 3.541e-1

ERA order 160, block 50x50:
  kernel L2 error = 1.045e-2
  state/direct A weighted error = 2.796e-3
  state/direct B weighted error = 1.265e-2
  state/direct memory-force error = 1.056e-1

ERA order 240, block 55x55:
  kernel L2 error = 6.521e-3
  state/direct A weighted error = 1.336e-3
  state/direct B weighted error = 6.563e-3
  state/direct memory-force error = 1.577e-2
```

Best current state-space validation output:

```text
results/time_domain/state_space_radiation_dense88_era240_b55/
```

Main figures:

```text
figures/state_space_kernel_norm.png
figures/state_space_ab_reconstruction.png
figures/state_space_memory_force_norm.png
```

Conclusion: ERA is suitable for the next adapter integration step, but it
should remain optional until order reduction and multi-sea-state validation are
complete.

Added an ERA order sweep with 55x55 Hankel blocks. The focused sweep gave:

```text
order 200: kernel = 4.974e-3, memory force = 4.790e-2
order 220: kernel = 4.805e-3, memory force = 2.557e-2
order 240: kernel = 6.521e-3, memory force = 1.577e-2
order 260: kernel = 6.833e-3, memory force = 1.207e-2
order 280: kernel = 7.278e-3, memory force = 1.144e-2
```

With thresholds `kernel <= 1.0e-2` and `memory force <= 2.0e-2`, the smallest
candidate is `ERA order = 240`.

The adapter-owned state-space response solver also ran the full dense-88
JONSWAP case against direct Cummins convolution:

```text
output = results/time_domain/state_space_response_dense88_era240_b55/
master displacement L2 error = 6.976e-2
master velocity L2 error = 4.543e-2
memory-force L2 error = 1.750e-2
memory-force RMS error = 8.086e-3
drift-slope error = 9.241e-2
centerline heave L2 error = 6.883e-3
centerline heave RMS error = 2.041e-3
```

DOF split:

```text
master heave L2/RMS error = 6.764e-3 / 2.023e-3
master pitch L2/RMS error = 1.574e-2 / 2.463e-3
surge L2 error = 5.582e-2, direct RMS norm = 5.078e1
sway L2 error = 7.112e-1, direct RMS norm = 3.002e0
roll L2 error = 6.106e-1, direct RMS norm = 7.671e-7
```

Interpretation: state-space radiation force is close, and centerline heave is
already within about 0.7% in time-history L2 and 0.2% in RMS. The full motion
norm is dominated by unconstrained/weakly constrained low-frequency horizontal
and rotational drift. ERA is therefore promising for hydroelastic heave
observables, but direct Cummins convolution remains the validated full-motion
baseline until mooring and passivity/drift-aware state-space validation are
added.

Closed-loop neighboring-order checks:

```text
ERA order 240: master = 6.976e-2, memory = 1.750e-2, heave = 6.883e-3
ERA order 260: master = 1.588e-1, memory = 1.631e-2, heave = 8.939e-3
ERA order 280: master = 1.466e-1, memory = 1.570e-2, heave = 7.610e-3
```

The production candidate remains `ERA order = 240` for this dense-88 JONSWAP
case because higher order lowers memory force slightly but worsens the
drift-sensitive closed-loop displacement response.

Added a simple four-corner spring mooring check. The mooring stiffness is
assembled in the adapter layer on corner nodes `(1, 61, 733, 793)` of the
61 x 13 structural grid and projected through the existing SEREP
transformation. It is not part of the RODM frequency-domain core.

ERA-240 mooring comparison:

```text
no mooring:
  master displacement error = 6.976e-2
  drift-slope error = 9.241e-2
  centerline heave error = 6.883e-3

four-corner horizontal springs, k = 1e6 N/m:
  master displacement error = 2.278e-2
  drift-slope error = 1.521e-2
  centerline heave error = 6.889e-3

four-corner horizontal springs, k = 1e7 N/m:
  master displacement error = 6.874e-3
  drift-slope error = 1.891e-3
  centerline heave error = 6.888e-3
```

This confirms that the previous large full-motion state-space error was mainly
a low-frequency stationkeeping issue. A simple mooring spring stabilizes the
closed-loop full response while preserving the validated heave agreement.

## Remaining To Do

1. Run the same numerical sensitivity check for several peak periods / sea
   states, not only the current `omega = 0.4157 rad/s` JONSWAP case.
2. Develop a multi-frequency residual or state-space radiation approximation
   for broader sea states.
3. Add optional output thinning or chunked writing for long time histories.
4. Add a documented benchmark report for the final validated Cummins settings.

## 2026-05-22 WEC-Sim-like Platform Wrapper

Added an external WEC-Sim-like platform entry point:

```text
src/offshore_energy_sim/time_domain_adapter/wecsim_like_solver.py
scripts/run_wecsim_like_time_domain_platform.py
```

The wrapper supports:

```text
direct Cummins convolution
ERA state-space radiation
state-space model save/load
wave-spectrum or regular-wave excitation
optional mooring linearization provider
adapter-owned arrays, metrics, figures, and reports
```

The mooring interface is intentionally a reduced linearization provider:

```text
provider(case, structural) -> MooringLinearization | ndarray | None
```

This leaves room for a future dedicated mooring module without coupling it into
the RODM frequency-domain core.

Main validation:

```text
output = results/time_domain/wecsim_like_platform_dm10/
hydrodynamics = DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
excitation = JONSWAP, Hs = 1.0 m, Tp = 15.1147 s
time samples = 2001
duration = 604.588 s
time step = 0.302294 s
ERA order = 240
Hankel blocks = 55 x 55
mooring = four-corner horizontal springs, 1e7 N/m per corner/DOF
```

Closed-loop state-space vs direct Cummins result:

```text
master displacement L2 error = 9.970e-3
master velocity L2 error = 1.424e-2
memory-force L2 error = 1.994e-2
global displacement L2 error = 9.846e-3
centerline heave L2 error = 9.764e-3
centerline heave RMS error = 3.185e-3
```

Full test result after the platform addition:

```text
79 passed in 2.64s
```

The implementation preserves the architecture boundary: RODM remains the
independent frequency-domain core, and WEC-Sim/Cummins/state-space logic remains
in `time_domain_adapter` and adapter validation scripts.

## 2026-05-22 Multi-Sea-State Validation

Added:

```text
scripts/validate_wecsim_like_multi_sea_state.py
docs/time_domain/wecsim_like_multi_sea_state_validation_2026_05_22.md
```

Scope preserved:

```text
RODM core modified = false
mooring module modified = false
adapter interfaces preserved = true
```

The run used six short sea states:

```text
Hs = 0.5, 1.0 m
omega_peak = 0.35, 0.4157, 0.55 rad/s
seed = 1
ERA order = 240
Hankel blocks = 55 x 55
short samples per case = 801
```

Short-case state-space vs direct Cummins envelope:

```text
max master displacement error = 1.080e-2
max memory-force error = 2.520e-2
max centerline heave RMS error = 4.816e-3
```

The long lightweight screening case used state-space only:

```text
Hs = 1.0 m
omega_peak = 0.4157 rad/s
duration = 1813.765 s
samples = 4801
reconstructed Hs = 0.9996 m
centerline heave RMS mean = 1.763e-1
output = metrics and figures only, no large arrays saved
```

Full test result after this addition:

```text
79 passed in 2.64s
```

## 2026-05-22 Frequency RMS Extension

Extended the multi-sea-state validation script to precompute a frequency-domain
centerline heave RAO and compare spectrum-integrated frequency RMS against
post-ramp time-domain RMS. The frequency comparison uses the same adapter-level
linear mooring stiffness; RODM core and the mooring module remain unchanged.

Expanded matrix:

```text
Hs = 1.0 m
omega_peak = 0.30, 0.35, 0.4157, 0.55, 0.70 rad/s
seeds = 1, 2
short cases = 10
samples per short case = 801
ERA order = 240
```

State-space vs direct Cummins:

```text
max master displacement error = 1.685e-2
max memory-force error = 5.587e-2
max centerline heave RMS error = 7.096e-3
```

Frequency-domain RMS comparison:

```text
frequency RAO omega points = 88
max frequency/direct time RMS error = 3.076e-1
max frequency/state time RMS error = 3.071e-1
```

The frequency/time RMS error is larger because these are short finite random
wave records; reconstructed `Hs` varies by seed. The state/direct comparison is
therefore the regression metric for short records, while theoretical
frequency-domain RMS closure should use longer records or ensemble averaging.

Long lightweight state-space screening:

```text
duration = 3627.531 s
samples = 9601
reconstructed Hs = 0.9912 m
centerline heave RMS mean = 0.1738
```

Full tests:

```text
79 passed in 2.69s
```

## 2026-05-22 Long Direct Frequency RMS Closure

Ran a long-record direct-Cummins closure check for two representative peak
frequencies and two random seeds:

```text
Hs = 1.0 m
omega_peak = 0.4157, 0.70 rad/s
seeds = 1, 2
cycles = 120
samples per case = 4801
cases = 4
ERA order = 240
```

State-space vs direct Cummins:

```text
max master displacement error = 2.333e-2
max memory-force error = 5.526e-2
max centerline heave RMS error = 4.102e-3
```

Frequency-domain RMS closure:

```text
max frequency/direct post-discard RMS error = 1.974e-2
max frequency/state post-discard RMS error = 2.081e-2
max frequency/direct fitted RMS error = 7.923e-3
max frequency/state fitted RMS error = 1.072e-2
```

Conclusion: the earlier short-record frequency/time RMS mismatch was dominated
by finite random-wave realization scatter. With 120 peak cycles, direct Cummins
closes against frequency-domain RMS below about `1%` using fitted harmonic
components, and ERA state-space remains close to the direct reference.

## 2026-05-22 RK4 Time Integrator

Added an optional explicit fourth-order Runge-Kutta integrator:

```text
solve_linear_time_domain_rk4(...)
solve_state_space_radiation_linear_system_rk4(...)
WecSimLikeRadiationConfig(integrator="newmark" | "rk4")
scripts/validate_time_integrator_comparison.py
```

The implementation keeps the same reduced-space workflow:

```text
1. assemble/reduce mass, stiffness, hydrostatic, hydrodynamic, and optional mooring matrices;
2. advance only reduced/master DOFs in time;
3. reconstruct global retained DOFs only for output.
```

Coarse RK4 check:

```text
cycles = 20
steps_per_cycle = 80
result = unstable / NaN
```

Fine-step RK4/Newmark comparison:

```text
cycles = 5
steps_per_cycle = 400
direct Cummins master error = 1.133e-4
direct Cummins memory error = 1.220e-4
direct Cummins heave RMS error = 3.318e-5
state-space master error = 1.110e-4
state-space memory error = 1.413e-4
state-space heave RMS error = 3.618e-5
```

Interpretation: RK4 is correct as a reduced-space explicit cross-check, but
Newmark remains the production default because the flexible reduced system is
stiff and explicit RK4 requires much smaller time steps.
