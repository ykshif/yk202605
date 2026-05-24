# RK4 Time Integrator Comparison - 2026-05-22

This note records the addition and validation of an explicit fourth-order
Runge-Kutta time integrator for the external WEC-Sim-like time-domain adapter.

The architecture boundary remains unchanged:

```text
RODM frequency-domain core modified = false
time integration occurs in reduced/master DOFs
global response is reconstructed only after reduced time stepping
```

## Implementation

Added:

```text
solve_linear_time_domain_rk4(...)
solve_state_space_radiation_linear_system_rk4(...)
scripts/validate_time_integrator_comparison.py
```

Extended:

```text
solve_linear_time_domain(..., integrator="newmark" | "rk4")
solve_state_space_radiation_linear_system(..., integrator="newmark" | "rk4")
WecSimLikeRadiationConfig(integrator="newmark" | "rk4")
scripts/run_wecsim_like_time_domain_platform.py --integrator newmark|rk4
scripts/validate_wecsim_like_multi_sea_state.py --integrator newmark|rk4
```

The RK4 path advances:

```text
qdot = v
vdot = M_eff^{-1} [F(t) - F_memory(t) - C_eff v - K_eff q]
```

where all matrices are already reduced:

```text
M_eff = M_reduced + A_inf + A_res
K_eff = K_struct_reduced + K_hs_reduced + K_moor_reduced
```

For Cummins radiation, the history force uses the same explicit known-history
convention as the Newmark path. The force is evaluated at grid points and
linearly interpolated within one RK4 step.

For ERA state-space radiation, the ERA radiation state remains a discrete
history realization at the grid points; the mechanical reduced system is
advanced with RK4 inside each time interval.

## Important Stability Observation

Newmark average acceleration is implicit and robust for the reduced flexible
system. RK4 is explicit and therefore has a time-step stability limit.

A coarse attempt with:

```text
cycles = 20
steps_per_cycle = 80
```

produced overflow/NaN in both direct Cummins and state-space paths. This is
expected for explicit integration of a stiff reduced hydroelastic system.

The stable comparison used:

```text
cycles = 5
steps_per_cycle = 400
memory_cycles = 2
ERA order = 240
four-corner horizontal mooring = 1e7 N/m per corner/DOF
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_integrator_comparison.py `
  --output-root results\time_domain\time_integrator_newmark_rk4_comparison_spc400 `
  --cycles 5 `
  --steps-per-cycle 400 `
  --memory-cycles 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

## Results

RK4 relative to Newmark:

| Radiation model | Master displacement | Memory force | Centerline heave L2 | Centerline heave RMS |
| --- | ---: | ---: | ---: | ---: |
| Direct Cummins | 1.133e-4 | 1.220e-4 | 6.646e-5 | 3.318e-5 |
| ERA state-space | 1.110e-4 | 1.413e-4 | 6.904e-5 | 3.618e-5 |

The fine-step RK4 result is essentially identical to Newmark for the observable
heave RMS response.

## Output

```text
results/time_domain/time_integrator_newmark_rk4_comparison_spc400/time_integrator_comparison_metrics.json
results/time_domain/time_integrator_newmark_rk4_comparison_spc400/figures/newmark_vs_rk4_centerline_heave_rms.png
```

## Interpretation

The RK4 implementation is correct as an optional explicit integrator, but it is
not the recommended production default for this flexible reduced system. Use:

```text
Newmark: production and long time histories
RK4: numerical cross-checks with sufficiently small time step
```

The current implementation satisfies the requested reduced-space workflow:

```text
1. reduce mass/stiffness/hydrodynamic matrices first;
2. integrate only reduced/master DOFs in time;
3. reconstruct global response only when output is required.
```
