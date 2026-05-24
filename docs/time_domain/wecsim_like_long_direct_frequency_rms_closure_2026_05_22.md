# WEC-Sim-like Long Direct Frequency RMS Closure - 2026-05-22

This note records the long-record frequency-domain RMS closure check requested
after the expanded short multi-sea-state matrix. The purpose is to separate
finite random-wave realization scatter from actual time-domain solver error.

Architecture boundary:

```text
RODM frequency-domain core modified = false
mooring module modified = false
validation remains in external adapter scripts
```

## Command

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py `
  --output-root results\time_domain\wecsim_like_long_direct_frequency_rms_closure `
  --hs-values 1.0 `
  --target-omega-values 0.4157,0.70 `
  --seeds 1,2 `
  --cycles 120 `
  --steps-per-cycle 40 `
  --memory-cycles 2 `
  --skip-long-run `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

## Case Matrix

```text
Hs = 1.0 m
omega_peak = 0.4157, 0.70 rad/s
seeds = 1, 2
cases = 4
samples per case = 4801
ERA order = 240
Hankel blocks = 55 x 55
frequency RAO points = 88
```

The `omega_peak = 0.4157 rad/s` cases cover the validated central sea state.
The `omega_peak = 0.70 rad/s` cases cover the higher-frequency point where the
expanded short matrix showed the largest memory-force error.

## State-Space vs Direct Cummins

Envelope:

```text
max state/direct master displacement error = 2.333e-2
max state/direct memory-force error = 5.526e-2
max state/direct centerline heave RMS error = 4.102e-3
```

Case-level heave RMS errors:

| Case | State/Direct Heave RMS Error |
| --- | ---: |
| omega 0.4157, seed 1 | 2.665e-3 |
| omega 0.4157, seed 2 | 2.674e-3 |
| omega 0.70, seed 1 | 3.883e-3 |
| omega 0.70, seed 2 | 4.102e-3 |

The heave RMS response remains within about `0.41%` of direct Cummins over
these long records.

## Frequency RMS Closure

Post-discard time-series RMS versus frequency-domain spectrum RMS:

```text
max frequency/direct time RMS error = 1.974e-2
max frequency/state time RMS error = 2.081e-2
```

Multi-harmonic fitted RMS versus frequency-domain spectrum RMS:

```text
max frequency/direct fitted RMS error = 7.923e-3
max frequency/state fitted RMS error = 1.072e-2
fitted RMS cases = 4
```

This is the key result. The earlier short-record frequency/time RMS mismatch
reached about `30%` because a finite random-wave realization with only 20 peak
cycles does not reproduce the theoretical spectrum RMS well. With 120 peak
cycles, the direct-Cummins frequency RMS closure falls below `1%` when using
the fitted harmonic components, and the state-space result remains close.

## Wave Realization Check

Reconstructed wave heights:

```text
omega 0.4157, seed 1: Hs = 0.9996 m
omega 0.4157, seed 2: Hs = 0.9950 m
omega 0.70, seed 1: Hs = 1.0024 m
omega 0.70, seed 2: Hs = 1.0194 m
```

The long records now reproduce the target `Hs = 1.0 m` well, which explains
the improved frequency/time RMS closure.

## Output Artifacts

```text
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/multi_sea_state_metrics.json
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/multi_sea_state_summary.csv
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/report.md
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/figures/state_space_direct_error_matrix.png
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/figures/centerline_heave_rms_summary.png
results/time_domain/wecsim_like_long_direct_frequency_rms_closure/figures/frequency_time_rms_errors.png
```

## Interpretation

The long direct-Cummins validation supports the current WEC-Sim-like platform:

```text
1. direct Cummins closes against frequency-domain RMS when the wave record is long enough;
2. ERA-240 state-space remains close to direct Cummins in heave RMS;
3. high-frequency sea states have larger memory-force error, but response RMS remains stable;
4. short records should be used for regression, not theoretical spectrum closure.
```

Next validation should use the long-record settings for a small accepted
benchmark set and the lightweight state-space path for broad sea-state scans.
