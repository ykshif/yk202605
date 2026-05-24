# Workflow Design

Date: 2026-04-28

## Purpose

This document describes the first standardized workflow layer for the offshore
energy simulation platform. The goal is to move from one-off research scripts
toward reusable case execution:

```text
load config -> build case -> solve -> save response -> validate -> plot -> report
```

Numerical-result expectation for this workflow refactor: no change.

## Standard Output Layout

The standard output root is:

```text
results/<case_id>/
```

The default variant writes:

```text
results/<case_id>/
├── response.npy
├── metrics.json
├── report.md
├── figures/
└── logs/
```

Explicit variants write:

```text
results/<case_id>/variants/<variant_id>/
├── response.npy
├── metrics.json
└── logs/
```

For the current 300 m case:

```text
results/reference_case_300/response.npy
results/reference_case_300/variants/hydro_reversed/response.npy
```

## Core Helpers

Workflow artifacts are managed by:

```text
src/offshore_energy_sim/core/workflow.py
```

Key API:

```text
WorkflowPaths
build_workflow_paths(case_root, variant_id="default")
write_metrics_json(path, metrics)
```

These helpers only manage paths and JSON serialization. They do not change any
hydroelastic numerical algorithm.

## Configuration

The current benchmark config is:

```text
configs/reference_case_300.yaml
```

It now supports both the original `inputs` section and newer platform-oriented
sections:

```text
hydrodynamics
structure
reduction
solver
validation
outputs
```

The reusable template is:

```text
configs/templates/rodm_frequency_case.yaml
```

An explicit provenance variant is:

```text
configs/reference_case_300_hydro_reversed.yaml
```

This variant enables `reverse_hydrodynamic_node_order: true`. It is not the
default legacy-equivalent path.

## Entrypoints

Generic config-driven runner:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_rodm_case_from_config.py --config configs\reference_case_300.yaml
```

The same runner now supports explicit domain selection:

```powershell
python scripts\run_rodm_case_from_config.py --config configs\reference_case_300.yaml --domain frequency
python scripts\run_rodm_case_from_config.py --config configs\reference_case_300.yaml --domain time --cycles 80 --steps-per-cycle 180
```

Frequency-domain runs keep writing the standard `response.npy`. Time-domain
runs write to a separate variant, for example:

```text
results/<case_id>/variants/time_domain/response.npy
results/<case_id>/variants/time_domain/time.npy
results/<case_id>/variants/time_domain/master_displacement_time.npy
```

The current time-domain path is a single-frequency linear validation model: it
uses the selected frequency's added mass and radiation damping as constant
coefficients, then fits the steady-state time response back to a complex
amplitude for comparison with the frequency-domain solver.

Explicit reversed variant:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_rodm_case_from_config.py --config configs\reference_case_300_hydro_reversed.yaml
```

Reference workflow with validation, plot, and report:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_reference_case_300_workflow.py
```

Full regression suite:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_refactor_regression_suite.py
```

Regular-wave batch validation for wavelengths 60 m, 120 m, 180 m, 240 m, and
300 m:

```powershell
D:\Users\KKKKK\anaconda3\Scripts\conda.exe run -n base python scripts\run_regular_wave_batch_validation.py
```

This writes:

```text
results/regular_wave_batch/
docs/regular_wave_batch_validation_report.md
```

## Validation Results

The standard workflow output has been checked against pre-refactor generated
outputs:

```text
standard_workflow_default_vs_pre_refactor
  max_abs_error: 0.0
  l2_relative_error: 0.0

standard_workflow_reversed_vs_pre_refactor
  max_abs_error: 0.0
  l2_relative_error: 0.0
```

The explicit reversed config has also been checked:

```text
hydro_reversed_config_vs_candidate
  max_abs_error: 0.0
  l2_relative_error: 0.0
```

## Next Design Step

The next platform step is to generalize the case-specific plotting and heave
extraction so additional wavelengths, wave directions, wind/PV cases, and
optimization runs can reuse the same workflow skeleton.
