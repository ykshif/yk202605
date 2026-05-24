# WEC-Sim-like Time-Domain Platform Status - 2026-05-22

This note records the first complete external WEC-Sim-like platform layer for
the RODM DM10 hydroelastic case. The RODM frequency-domain solver remains the
main independent method. The time-domain platform reads RODM-compatible inputs
and runs as an adapter layer.

## Architecture Boundary

The implemented boundary is:

```text
RODM frequency-domain inputs/results
  -> time_domain_adapter WEC-Sim-like platform
  -> Cummins direct convolution or ERA state-space radiation
  -> optional external mooring linearization provider
  -> time histories, arrays, metrics, figures
```

The dependency direction is one-way. RODM core code does not import or depend
on `time_domain_adapter`.

## Added Platform API

Added:

```text
src/offshore_energy_sim/time_domain_adapter/wecsim_like_solver.py
scripts/run_wecsim_like_time_domain_platform.py
```

Extended:

```text
src/offshore_energy_sim/time_domain_adapter/state_space_radiation.py
src/offshore_energy_sim/time_domain_adapter/__init__.py
tests/test_state_space_radiation.py
tests/test_wecsim_like_adapter.py
```

The platform entry point is:

```python
solve_rodm_wecsim_like_time_domain(case, config, radiation=..., mooring_provider=...)
```

The radiation option is controlled by:

```python
WecSimLikeRadiationConfig(model="direct_convolution")
WecSimLikeRadiationConfig(model="state_space", state_space_order=240)
```

The mooring interface is:

```python
MooringLinearization(reduced_stiffness, metadata={...})
```

or a provider callback:

```python
provider(case, structural) -> MooringLinearization | ndarray | None
```

This keeps mooring outside both the RODM frequency-domain core and the
hydrodynamic radiation model. The four-corner spring is only an adapter
validation example.

## State-Space Model Persistence

ERA radiation models can now be persisted and reused:

```python
save_discrete_state_space_radiation_model(model, path)
load_discrete_state_space_radiation_model(path)
```

The archive stores:

```text
state_matrix
input_matrix
output_matrix
zero_lag_kernel
time_step
fit_l2_relative_error
spectral_radius
```

This is needed for long sea-state sweeps because the fitted radiation model can
be reused when the time step and reduced DOF layout are unchanged.

## Platform Validation Command

The main validation run used the dense-88 DM10 mesh2 Cummins hydrodynamic data:

```powershell
.\.venv\Scripts\python.exe scripts\run_wecsim_like_time_domain_platform.py `
  --output-root results\time_domain\wecsim_like_platform_dm10 `
  --cycles 40 `
  --steps-per-cycle 50 `
  --memory-cycles 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000 `
  --save-state-space-model-path results\time_domain\wecsim_like_platform_dm10\state_space_radiation_era240.npz `
  --save-arrays
```

The default input is:

```text
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
```

The sea-state input is:

```text
JONSWAP spectrum
Hs = 1.0 m
Tp = 15.1147 s
omega_peak = 0.4157 rad/s
gamma = 3.3
seed = 1
duration = 604.588 s
time step = 0.302294 s
time samples = 2001
```

The mooring example is:

```text
four corner horizontal springs
corner nodes = 1, 61, 733, 793
k_surge = k_sway = 1.0e7 N/m per corner
k_heave = 0
```

## Validation Results

Direct Cummins and ERA-240 state-space results:

```text
ERA model order = 240
ERA Hankel blocks = 55 x 55
ERA kernel fit L2 relative error = 6.535e-3
ERA spectral radius = 0.999
```

Closed-loop response comparison:

```text
state/direct master displacement L2 error = 9.970e-3
state/direct master velocity L2 error = 1.424e-2
state/direct radiation memory force L2 error = 1.994e-2
state/direct global displacement L2 error = 9.846e-3
state/direct centerline heave L2 error = 9.764e-3
state/direct centerline heave RMS error = 3.185e-3
```

RMS levels:

```text
direct master displacement RMS norm = 6.122e-1
state master displacement RMS norm = 6.119e-1
direct memory force RMS norm = 6.153e6
state memory force RMS norm = 6.142e6
direct centerline heave RMS mean = 1.924e-1
state centerline heave RMS mean = 1.923e-1
reconstructed Hs from wave elevation = 1.081 m
```

The state-space result is now close enough to direct Cummins for this baseline
linear hydroelastic sea-state calculation. Direct Cummins remains the reference
method, while ERA state-space is the practical WEC-Sim-like radiation option.

## Output Artifacts

```text
results/time_domain/wecsim_like_platform_dm10/wecsim_like_platform_metrics.json
results/time_domain/wecsim_like_platform_dm10/report.md
results/time_domain/wecsim_like_platform_dm10/wecsim_like_platform_summary.csv
results/time_domain/wecsim_like_platform_dm10/state_space_radiation_era240.npz
results/time_domain/wecsim_like_platform_dm10/arrays/
results/time_domain/wecsim_like_platform_dm10/figures/
```

Important figures:

```text
figures/direct_convolution_representative_centerline_heave.png
figures/state_space_representative_centerline_heave.png
figures/radiation_memory_force_norm.png
figures/direct_vs_state_centerline_heave_time.png
figures/direct_vs_state_centerline_heave_rms.png
```

## Tests

Current full test result:

```text
79 passed in 2.64s
```

Covered items include:

```text
hydrodynamic extrapolation
radiation kernel construction
Cummins time-domain helpers
ERA state-space radiation
state-space model save/load
mooring adapter projection
WEC-Sim-like platform config and mooring interface checks
```

## Current Completion Level

Complete for the current linear WEC-Sim-like platform stage:

```text
wave spectrum or regular wave input
frequency-domain excitation synthesis
Cummins direct-convolution radiation
ERA state-space radiation approximation
RODM structural reduction coupling
optional reduced linear mooring provider
time histories for master and global DOFs
centerline heave extraction
metrics, arrays, figures, reports
state-space model save/load
```

Still intentionally left for later specialized modules:

```text
nonlinear mooring line dynamics
PTO/control models
nonlinear hydrodynamic corrections
multi-body contact/constraints
adaptive or implicit radiation-state integration
large sea-state batch orchestration
formal comparison with external WEC-Sim cases
```

The next technically appropriate stage is a sea-state sweep using the saved
ERA-240 model, followed by a formal mooring module interface once the dedicated
mooring implementation is available.
