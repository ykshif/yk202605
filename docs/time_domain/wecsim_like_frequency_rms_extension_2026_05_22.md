# WEC-Sim-like Frequency RMS Extension - 2026-05-22

This note records the expanded validation after the first multi-sea-state
matrix. The goal was to add a frequency-domain RMS reference while keeping the
same architecture:

```text
RODM frequency-domain core unchanged
mooring module unchanged
time-domain validation remains in adapter scripts
```

## Script Update

Extended:

```text
scripts/validate_wecsim_like_multi_sea_state.py
```

New behavior:

```text
1. precompute centerline heave RAO on the full 88-point hydrodynamic omega grid;
2. include the same adapter-level reduced linear mooring stiffness in the frequency-domain comparison;
3. compute frequency-domain centerline heave RMS by spectrum integration;
4. compare frequency RMS with post-ramp direct Cummins and ERA state-space time RMS;
5. skip multi-harmonic time fitting when the short time record is underdetermined or likely ill-conditioned.
```

The multi-harmonic fit was initially tested, but short 20-cycle records with
88 frequency components are not long enough for a well-conditioned fit. The
script now marks these cases as:

```text
fit_status = skipped_insufficient_post_discard_samples
```

This prevents nonphysical fitted RMS values from being used as validation
evidence.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py `
  --output-root results\time_domain\wecsim_like_multi_sea_state_frequency_rms_extended_clean `
  --hs-values 1.0 `
  --target-omega-values 0.30,0.35,0.4157,0.55,0.70 `
  --seeds 1,2 `
  --cycles 20 `
  --steps-per-cycle 40 `
  --memory-cycles 2 `
  --long-cycles 240 `
  --long-target-omega 0.4157 `
  --long-hs 1.0 `
  --long-seed 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

## State-Space vs Direct Cummins

The expanded matrix has:

```text
Hs = 1.0 m
omega_peak = 0.30, 0.35, 0.4157, 0.55, 0.70 rad/s
seeds = 1, 2
short cases = 10
samples per short case = 801
ERA order = 240
Hankel blocks = 55 x 55
```

Envelope:

```text
max state/direct master displacement error = 1.685e-2
max state/direct memory-force error = 5.587e-2
max state/direct centerline heave RMS error = 7.096e-3
```

The heave RMS agreement remains below `0.71%` over the expanded peak-frequency
and seed matrix. The largest memory-force relative error occurs at the high
peak-frequency case (`omega_peak = 0.70 rad/s`), but the heave RMS response
remains close.

## Frequency RMS Comparison

Frequency-domain RAO precompute:

```text
omega points = 88
elapsed = 5.954 s
```

Post-discard time-series RMS compared with frequency-domain spectrum RMS:

```text
max frequency/direct time RMS error = 3.076e-1
max frequency/state time RMS error = 3.071e-1
```

These larger errors are not interpreted as a Cummins/state-space failure. They
mainly reflect finite-duration random-wave realization effects in the 20-cycle
short records. For example, reconstructed short-record wave heights vary from
about `0.821 m` to `1.138 m` even though the target spectrum has `Hs = 1.0 m`.
The frequency RMS is the theoretical spectrum integral, while the time RMS is
estimated from a short finite realization after ramp removal.

The important observation is that direct Cummins and ERA state-space track each
other closely under the same finite realization. For theoretical RMS closure,
the validation should use longer time records or ensemble averaging over more
seeds.

## Long Lightweight Screening

The long state-space-only screening was extended to 240 peak cycles:

```text
case = Hs1_om0p4157_seed2
duration = 3627.531 s
time samples = 9601
time step = 0.377868 s
reconstructed Hs = 0.9912 m
centerline heave RMS mean = 0.1738
```

Six-window checks:

```text
wave Hs windows = [0.956, 0.964, 1.061, 1.061, 0.927, 0.969] m
mean centerline heave RMS windows = [0.167, 0.167, 0.188, 0.191, 0.159, 0.168]
```

No numerical drift or blow-up was observed in this long lightweight run.

## Output Artifacts

```text
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/multi_sea_state_metrics.json
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/multi_sea_state_summary.csv
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/report.md
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/figures/state_space_direct_error_matrix.png
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/figures/centerline_heave_rms_summary.png
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/figures/frequency_time_rms_errors.png
results/time_domain/wecsim_like_multi_sea_state_frequency_rms_extended_clean/figures/long_lightweight_window_metrics.png
```

## Test Result

```text
79 passed in 2.69s
```

## Next Step

For frequency-domain RMS closure, use one of these strategies:

```text
1. longer direct-Cummins validation records for a small number of sea states;
2. ensemble averaging over several random seeds;
3. keep short records for state/direct regression only, not theoretical RMS closure;
4. keep state-space-only long screening for broad sea-state scans.
```
