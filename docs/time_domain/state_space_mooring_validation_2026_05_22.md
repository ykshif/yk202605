# State-Space Mooring Validation - 2026-05-22

This note records a simple four-corner linear-spring mooring validation for the
external RODM time-domain adapter. The RODM frequency-domain core remains
unchanged; the mooring stiffness is assembled and applied only inside the
adapter validation workflow.

## Implementation

Added:

```text
src/offshore_energy_sim/time_domain_adapter/mooring.py
tests/test_mooring_adapter.py
```

Extended:

```text
scripts/validate_state_space_response.py
```

The mooring model is intentionally simple:

```text
four corner nodes on the 61 x 13 FEM grid:
  1, 61, 733, 793

spring DOFs:
  retained DOF 0 = surge
  retained DOF 1 = sway

default validation:
  no vertical spring
  no nonlinear line geometry
  no pretension
```

The global retained-DOF diagonal spring stiffness is projected into the RODM
master coordinates using the existing SEREP transformation:

```text
K_moor_reduced = T^T K_moor_global_disordered T
```

This keeps mooring external to the RODM frequency-domain assembly.

## Commands

Baseline without mooring:

```powershell
.\.venv\Scripts\python.exe scripts\validate_state_space_response.py `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --output-root results\time_domain\state_space_response_dense88_era240_b55
```

Four-corner horizontal spring, 1e6 N/m per corner and horizontal DOF:

```powershell
.\.venv\Scripts\python.exe scripts\validate_state_space_response.py `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 1e6 `
  --output-root results\time_domain\state_space_response_dense88_era240_b55_moor_k1e6
```

Four-corner horizontal spring, 1e7 N/m:

```powershell
.\.venv\Scripts\python.exe scripts\validate_state_space_response.py `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 1e7 `
  --output-root results\time_domain\state_space_response_dense88_era240_b55_moor_k1e7
```

## Results

| Mooring k per corner/DOF | Master Disp. Error | Velocity Error | Memory-Force Error | Drift-Slope Error | Centerline Heave Error | Heave RMS Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.976e-2 | 4.543e-2 | 1.750e-2 | 9.241e-2 | 6.883e-3 | 2.041e-3 |
| 1e6 N/m | 2.278e-2 | 3.527e-2 | 1.758e-2 | 1.521e-2 | 6.889e-3 | 2.054e-3 |
| 1e7 N/m | 6.874e-3 | 2.885e-2 | 1.755e-2 | 1.891e-3 | 6.888e-3 | 2.045e-3 |

The simple mooring stiffness suppresses the drift-sensitive full-motion error
without degrading the hydroelastic heave response. The best tested value is
`1e7 N/m` per corner in surge and sway.

## Interpretation

The earlier no-mooring closed-loop comparison was dominated by weakly
constrained surge/sway/roll drift. Adding even a simple four-corner horizontal
spring changes the validation picture:

- full master displacement error drops from `6.976e-2` to `6.874e-3`;
- drift-slope error drops from `9.241e-2` to `1.891e-3`;
- centerline heave remains stable at about `6.9e-3` L2 error;
- radiation-memory force error remains about `1.75e-2`.

This supports the physical interpretation that the ERA state-space radiation
model is already accurate for heave-dominated hydroelastic observables, and the
large no-mooring full-motion error was primarily a low-frequency stationkeeping
issue.

## Figures

```text
results/time_domain/state_space_response_dense88_era240_b55_moor_k1e7/figures/state_space_vs_direct_master_displacement_norm.png
results/time_domain/state_space_response_dense88_era240_b55_moor_k1e7/figures/state_space_vs_direct_memory_force_norm.png
results/time_domain/state_space_response_dense88_era240_b55_moor_k1e7/figures/state_space_vs_direct_centerline_heave_rms.png
results/time_domain/state_space_response_dense88_era240_b55_moor_k1e7/figures/state_space_vs_direct_centerline_heave_time.png
```

## Next Step

The current four-corner spring model is enough for validation. The next
engineering step is to replace it with a configurable mooring matrix or a
linearized line model, then repeat the sea-state sweep before exposing
`radiation_model = "state_space"` as a normal adapter option.

