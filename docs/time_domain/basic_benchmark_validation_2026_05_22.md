# Basic Time-Domain Benchmark Validation

Date: 2026-05-22

## Purpose

Before moving to the next development stage, the current time-domain layer is
validated with a basic benchmark matrix:

```text
SDOF analytic/frequency-domain check
RODM regular-wave constant A/B check
RODM Cummins direct-convolution check
hydrodynamic reconstruction check
JONSWAP-DM10 mesh2 spectrum statistics check
adapter extrapolation and radiation-kernel stability check
```

This validation keeps the architecture boundary:

```text
RODM frequency-domain model remains the main independent solver.
WEC-Sim/Cummins-style time-domain logic remains in external adapter/scripts.
```

## Command

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_domain_basic_benchmark.py
```

Main output:

```text
results/time_domain/basic_benchmark_validation/basic_benchmark_metrics.json
results/time_domain/basic_benchmark_validation/figures/basic_benchmark_dashboard.png
```

## Validation Results

The benchmark status is `passed`.

```text
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

All rows satisfy the current acceptance targets.

## Figures

```text
results/time_domain/basic_benchmark_validation/figures/sdof_frequency_vs_time_validation.png
results/time_domain/basic_benchmark_validation/figures/validation_error_summary.png
results/time_domain/basic_benchmark_validation/figures/spectrum_closure_before_after.png
results/time_domain/basic_benchmark_validation/figures/basic_benchmark_dashboard.png
```

## Interpretation

The SDOF check verifies the Newmark integrator, force convention, and harmonic
amplitude fitting. The RODM regular-wave checks verify that the time-domain
layer can reproduce the existing frequency-domain solution. The Cummins check
verifies that the radiation-memory path is consistent with the selected
frequency response when the finite-band residual correction is enabled. The
JONSWAP statistics checks verify the spectrum-to-force-to-response chain. The
adapter extrapolation check verifies that original hydrodynamic data remain
unchanged while the radiation-kernel tail becomes more stable.

This benchmark is now the recommended gate before testing state-space
radiation approximation or other WEC-Sim-like extensions.
