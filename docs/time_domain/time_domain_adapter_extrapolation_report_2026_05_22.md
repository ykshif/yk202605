# RODM Time-Domain Adapter Extrapolation Report

Date: 2026-05-22

## Architecture Boundary

This step keeps the architecture boundary explicit:

```text
RODM frequency-domain hydroelastic solver
  -> exported hydrodynamic/frequency-domain data
  -> offshore_energy_sim.time_domain_adapter
  -> Cummins/WEC-Sim-like time-domain diagnostics and validation
```

No RODM frequency-domain core algorithm was modified in this step. The adapter
generates a new extrapolated hydrodynamic dataset and validation outputs. It
does not overwrite the original BEM/RODM data.

## Added Adapter Modules

```text
src/offshore_energy_sim/time_domain_adapter/__init__.py
src/offshore_energy_sim/time_domain_adapter/hydrodynamic_extrapolation.py
src/offshore_energy_sim/time_domain_adapter/radiation_kernel.py
src/offshore_energy_sim/time_domain_adapter/cummins_solver.py
src/offshore_energy_sim/time_domain_adapter/wecsim_like_validation.py
src/offshore_energy_sim/time_domain_adapter/extrapolation_diagnostics.py
scripts/validate_hydrodynamic_extrapolation.py
tests/test_hydrodynamic_extrapolation.py
tests/test_radiation_kernel.py
```

## Extrapolation Case

Source hydrodynamic dataset:

```text
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc
```

Adapter-generated dataset:

```text
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/hydrodynamics/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2_adapter_extrapolated.nc
```

Extrapolation settings:

```text
original omega range = 0.10 to 2.00 rad/s, 42 frequencies
extended omega range = 0.02 to 8.00 rad/s, 142 frequencies
low-frequency extension = 4 points
high-frequency extension = 96 points
B(omega) low-frequency behavior = ramp toward zero
B(omega) high-frequency behavior = power-law decay with cosine taper to zero
A(omega) high-frequency behavior = tail-limit continuation
F_ex(omega) high-frequency behavior = power-law decay with cosine taper
```

## Original-Range Protection

Maximum absolute differences inside the original frequency range:

```text
added_mass = 0.0
radiation_damping = 0.0
wave_force = 0.0
Froude_Krylov_force = 0.0
diffraction_force = 0.0
excitation_force = 0.0
```

This confirms that the adapter did not alter the original hydrodynamic data.

## Radiation Kernel Comparison

The Cummins radiation kernel was built from the original and extrapolated
`B(omega)` data using the same memory grid:

```text
selected omega = 0.4157 rad/s
kernel time samples = 161
memory length = 4 peak periods
passivity correction = clip_negative_eigenvalues
```

Key diagnostics:

```text
tail_rms_to_peak_ratio before = 3.694e-3
tail_rms_to_peak_ratio after  = 2.853e-3
after / before                = 0.772

tail_peak_to_peak_ratio before = 7.357e-3
tail_peak_to_peak_ratio after  = 5.638e-3
after / before                 = 0.766

norm_oscillation_score before = 0.472
norm_oscillation_score after  = 0.358

memory_integral_frobenius_norm before = 9.932e6
memory_integral_frobenius_norm after  = 9.268e6
```

The extended high-frequency tail reduces the normalized long-time kernel tail
and lowers the kernel norm oscillation score.

Diagnostic figures:

```text
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/figures/radiation_kernel_before_extrapolation.png
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/figures/radiation_kernel_after_extrapolation.png
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/figures/hydrodynamic_A_B_comparison.png
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/figures/excitation_force_extrapolation_comparison.png
```

## JONSWAP-DM10 Mesh2 Time-Domain Comparison

Both cases used:

```text
H_s = 1.0 m
spectrum = JONSWAP
time_samples = 3201
duration = 1209.176869 s
discard = 5 peak periods
radiation_convolution_rule = trapezoidal
```

Comparison:

```text
wave_variance_closure_error:
  before = 1.991e-2
  after  = 6.708e-3
  after / before = 0.337

excitation_force_rms_closure_error:
  before = 1.144e-2
  after  = 4.007e-3
  after / before = 0.350

centerline_heave_rms_closure_error:
  before = 1.594e-2
  after  = 3.358e-3
  after / before = 0.211

centerline_heave_rms_max:
  before = 0.270173
  after  = 0.268083
  after / before = 0.992

radiation_force_rms:
  before = 6.074e6
  after  = 6.210e6
  after / before = 1.022

heave_mean_drift_over_rms_l2:
  before = 4.148e-2
  after  = 4.797e-2
  after / before = 1.157
```

The extrapolated case improves wave, excitation-force, and response RMS closure
while preserving nearly the same response RMS level. The simple drift indicator
increases slightly, so drift/window sensitivity remains a follow-up item.

## Validation Command

```powershell
.\.venv\Scripts\python.exe scripts\validate_hydrodynamic_extrapolation.py --output-root results\time_domain\hydrodynamic_extrapolation_dm10_mesh2 --run-time-domain-comparison
```

Main metrics file:

```text
results/time_domain/hydrodynamic_extrapolation_dm10_mesh2/hydrodynamic_extrapolation_metrics.json
```

## Current Conclusion

The low/high-frequency extrapolation is now an opt-in adapter operation. It
does not modify the original RODM hydrodynamic dataset or the RODM
frequency-domain solver. For this DM10 mesh2 case, the tuned high-frequency
tail improves the Cummins radiation-kernel tail metrics and improves the
long JONSWAP time-domain RMS closure.

The next suitable stage is state-space radiation approximation, but only after
one more sensitivity sweep over high-frequency tail length, memory duration,
and time step.
