# hydrodynamics

Capytaine hydrodynamic data generation, loading, and hydrodynamic matrix/force access.

## Capytaine rectangular-array generator

`capytaine_array.py` is the standardized hydrodynamic entry point for the RODM
multi-floating-body workflow. It builds a rectangular module grid, keeps the
immersed mesh under the free surface, assigns six rigid-body DOFs per module,
runs radiation and diffraction problems for the requested angular frequencies
and wave directions, then writes a Capytaine-style `.nc` file with separated
complex values. It also computes Capytaine's Response Amplitude Operator (RAO)
from the assembled hydrodynamic dataset and stores it as the optional `rao`
variable for visualization and response checks.

The DOF labels follow the existing RODM datasets:

```text
0_0__Surge, 0_0__Sway, 0_0__Heave, 0_0__Roll, 0_0__Pitch, 0_0__Yaw, ...
```

RODM response preparation can then load the generated file through
`open_hydrodynamic_dataset()` and remove yaw/rz in the usual retained-DOF step.
