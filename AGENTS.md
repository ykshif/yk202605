# AGENTS.md

Guidance for future Codex work in this repository.

## Repository Context

This repository contains research code for offshore floating photovoltaic hydroelastic analysis. The long-term goal is to refactor the current scripts and notebooks into an integrated simulation platform for offshore wind and offshore floating photovoltaic systems, including:

- displacement response;
- stress and strength analysis;
- hydroelastic response;
- wind-wave coupled loading;
- power generation and power-loss analysis;
- future design optimization.

The current codebase includes paper-reproduction notebooks, exploratory research scripts, generated data files, and numerical kernels. Treat it as a research archive plus an evolving platform.

## Non-Negotiable Rules

1. Do not change numerical algorithms unless explicitly requested.
2. Preserve all original paper-reproduction scripts.
3. Do not delete data files.
4. Prefer small, isolated refactoring changes.
5. Every change must explain whether numerical results are expected to change.
6. Add comments for matrix dimensions and physical meaning when touching numerical code.
7. Keep old scripts runnable during refactoring.
8. Do not modify unrelated files while completing a focused task.

## Numerical Safety

Before changing numerical code, identify:

- matrix dimensions;
- DOF convention, especially 6-DOF versus reduced 5-DOF forms;
- node numbering convention, especially one-based Abaqus node IDs versus zero-based Python indices;
- physical meaning of each matrix or vector, such as mass, damping, stiffness, excitation force, displacement, stress, wind load, wave spectrum, or transformation matrix;
- expected output shape and units where known.

For every numerical change, state one of:

- numerical results are expected to be unchanged;
- numerical results may change only due to formatting, precision, or sparse/dense implementation details;
- numerical results are intentionally changed, with the reason;
- numerical impact is unknown and needs validation.

Prefer adding validation checks before changing formulas.

## Refactoring Style

Use small, reversible steps.

Good first changes:

- add documentation;
- add configuration wrappers;
- add tests around existing behavior;
- extract duplicated code without changing formulas;
- add shape and dimension comments;
- add path configuration while preserving old defaults;
- create new package modules that call existing functions.

Avoid early changes that:

- rewrite SEREP, condensation, or solver math;
- alter DOF ordering;
- alter master-node ordering;
- change force scaling;
- change stiffness, mooring, hinge, or damping constants;
- delete notebooks or generated results;
- move files in a way that breaks old scripts.

## Preserve Existing Workflows

Old scripts and notebooks must remain runnable during refactoring. If new code replaces an old path, keep a compatibility wrapper or document the old command.

Paper-reproduction notebooks and scripts are part of the research record. Do not delete or rewrite them as cleanup. If organization is needed, move copies only after explicit approval and preserve the originals.

Generated files such as `.npy`, `.inp`, `.mtx`, `.nc`, `.txt`, `.csv`, `.pdf`, `.avi`, or `.zip` should not be deleted unless the user explicitly asks.

## Comments And Documentation

When touching numerical code, add concise comments for:

- matrix shape, for example `(num_nodes * 5, num_nodes * 5)`;
- vector shape, for example `(1, num_master_nodes * 5)`;
- physical meaning, for example added mass, radiation damping, hydrostatic stiffness, structural stiffness, wind damping, wave excitation force;
- coordinate or DOF convention;
- whether a value is empirical, calibrated, or temporary.

Do not add noisy comments that merely restate the code.

## Target Architecture

New code should gradually move toward this package structure:

```text
src/offshore_energy_sim/
├── core/
├── geometry/
├── environment/
├── hydrodynamics/
├── structure/
├── reduction/
├── solver/
├── loads/
├── response/
├── strength/
├── power/
├── optimization/
├── postprocess/
└── utils/
```

Suggested module responsibilities:

- `core/`: shared case definitions, units, constants, and simulation orchestration.
- `geometry/`: mesh layout, node selection, grid/module geometry, coordinate helpers.
- `environment/`: sea states, wind states, solar conditions, environmental cases.
- `hydrodynamics/`: Capytaine dataset loading, added mass, damping, hydrostatic stiffness, wave excitation.
- `structure/`: Abaqus matrix readers, structural mass/stiffness, mooring, hinge/interconnection stiffness.
- `reduction/`: DOF reduction, SEREP, static condensation, dynamic condensation, reconstruction.
- `solver/`: frequency-domain and future time-domain solvers.
- `loads/`: wind load, wave load, combined environmental loads.
- `response/`: displacement, velocity, acceleration, and response spectrum calculations.
- `strength/`: internal force, stress, strength checks, fatigue placeholders.
- `power/`: PV generation, tilt/shading/response-induced power loss.
- `optimization/`: future design optimization and parameter studies.
- `postprocess/`: plots, validation figures, result export, animations.
- `utils/`: general helpers that do not belong to a physical domain.

## Recommended Work Pattern

For each task:

1. Read the relevant existing script or notebook first.
2. Identify whether the task is documentation, refactoring, bug fixing, validation, or new feature work.
3. Keep the change scoped to the requested area.
4. Preserve old entry points unless the user explicitly asks to replace them.
5. Run a lightweight verification when possible.
6. In the final summary, state:
   - files changed;
   - whether Python source code changed;
   - whether numerical results are expected to change;
   - what verification was performed.

## High-Risk Areas

Be especially careful with:

- `SEREP.py`;
- `DM_Assemble.py`;
- `DM_Reading.py`;
- `DM_Method.py`;
- `RODM_Wind_main.py`;
- hinge/interconnection stiffness assembly;
- Abaqus `.mtx` parsing;
- master/slave DOF ordering;
- wave and wind force scaling;
- generated Abaqus boundary-condition files.

Changes in these areas should be small and validated against reference outputs.

