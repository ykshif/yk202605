# Reference Cases

This file records baseline cases that should be preserved during refactoring. The purpose is to keep a stable comparison target before moving code into the future `src/offshore_energy_sim/` package.

## Baseline Case 1: 300 m x 60 m Floating Body, Wavelength 300 m

### Why This Case

This is the best first baseline case found in the current repository because it has:

- a 300 m by 60 m floating body validation context;
- hydrodynamic data for wavelength 300 m;
- structural mass and stiffness matrices for the 5 m x 5 m mesh;
- existing local computed displacement result;
- experimental heave RAO data;
- Fu et al. numerical simulation data for comparison;
- notebook history showing mesh-check and paper-comparison workflows.

This case should be used as a numerical guardrail before refactoring `SEREP.py`, `DM_Assemble.py`, `DM_Method.py`, or any DOF ordering logic.

### Source Notebooks

Primary notebooks containing this case:

- `RODM_Static_checkmesh.ipynb`
  - Uses `DM10_300_direction0.nc`.
  - Compares coarse, 5 m x 5 m, and finer mesh results.
  - Saves `displacement_1530mesh_300.npy`.
  - Contains plotting against `exp_300.txt`.

- `FEM_Reduce_v8 copy.ipynb`
  - Explicitly states comparison with Fu et al. numerical results.
  - Uses `number = 300`.
  - Uses `DM10_300_direction0.nc`.
  - Uses `JobMesh5_5_MASS1.mtx` and `JobMesh5_5_STIF1.mtx`.
  - Compares present heave RAO with `exp_300.txt` and `fu_sim300.txt`.

- `FEM_Reduce_v8_laptop.ipynb`
  - Similar Fu et al. comparison workflow.
  - Also references `exp_300.txt` and `fu_sim300.txt`.

### Input Files

Hydrodynamic data:

```text
E:\phd\Code\DM-FEM2D\HydrodynamicData\Yoga\DM10_300_direction0.nc
```

Structural matrices:

```text
E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_MASS1.mtx
E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_STIF1.mtx
```

Experimental and comparison data:

```text
E:\phd\Code\DM-FEM2D\data\Experiment_300_60\exp_300.txt
E:\phd\Code\DM-FEM2D\data\Experiment_300_60\fu_sim300.txt
```

Existing local output used as the current baseline:

```text
displacement_55mesh_300.npy
```

### Current Availability

All required input files above exist on this machine as of 2026-04-27.

The local result file `displacement_55mesh_300.npy` also exists in this repository.

### Model Parameters From Notebook

From `FEM_Reduce_v8 copy.ipynb`:

```python
number = 300
num_nodes = 793
master_nodes = DM_A.calculate_node_positions(424, 6, 10)
nodes_per_row = 61
Area = 5 * 5
removed_dof = [5]  # zero-based 6th DOF
mass_matrix = "JobMesh5_5_MASS1.mtx"
stiffness_matrix = "JobMesh5_5_STIF1.mtx"
hydrodynamic_dataset = "DM10_300_direction0.nc"
```

The workflow uses:

```python
M_consistant = SEREP.reduce_dofs(M, num_nodes, [5])
k = SEREP.reduce_dofs(k, num_nodes, [5])
M = SEREP.transform_mass_matrix(M_consistant, beta=0)
MasterDofs, SlaveDofs = SEREP.separate_dofs(num_nodes, master_nodes)
MR, KR, T = SEREP.SEREP(k, M, SlaveDofs, master_nodes)
```

Then it solves:

```text
(-omega^2 M - i omega C + K) X = F
```

using `DM_Assemble.solve_frequency_domain`.

### Response Extraction

For the 5 m x 5 m mesh result, the current notebook extracts the centerline heave response as:

```python
response = np.load("displacement_55mesh_300.npy")
mid = response[367*5-5:427*5-5, :]
heave = abs(mid[2::5])
x_present = np.linspace(0, 1, 60)
```

Interpretation:

- `response.shape = (3965, 1) = (793 nodes * 5 DOF, 1)`;
- `mid` selects the centerline node band used for validation;
- `heave` selects the 3rd retained DOF, interpreted as vertical displacement/heave RAO;
- `x_present` is the normalized longitudinal coordinate `x/L`.

### Current Numerical Baseline

Computed from the existing local file `displacement_55mesh_300.npy`:

```text
response shape:       (3965, 1)
response dtype:       complex128
heave length:         60
heave abs min:        0.8166492461156475
heave abs max:        1.2525429563334871
heave abs mean:       0.896981533364751
heave abs L2 norm:    7.000919304253492

heave first 5:
  1.2525429563334871
  1.188277994775755
  1.1278205922804936
  1.0734918039845933
  1.0250088407220588

heave last 5:
  1.0087753711864196
  1.0546212300261215
  1.1045626624034566
  1.1602294991098687
  1.2218692365109398
```

Comparison data sizes:

```text
exp_300.txt:     9 points
fu_sim300.txt:   107 points
```

Simple interpolation-based comparison using the current heave response:

```text
RMSE versus exp_300.txt:      0.06367482251124734
RMSE versus fu_sim300.txt:    0.04488934895346538
```

These RMSE values are not yet formal acceptance criteria. They are initial reference numbers to detect accidental numerical drift during refactoring.

### Structural Matrix Metadata

Mass matrix:

```text
file:       JobMesh5_5_MASS1.mtx
rows:       19518
max node:   793
full DOFs:  4758 = 793 * 6
```

Stiffness matrix:

```text
file:       JobMesh5_5_STIF1.mtx
rows:       51944
max node:   793
full DOFs:  4758 = 793 * 6
```

After removing DOF `[5]`, the retained structural system has:

```text
793 * 5 = 3965 DOFs
```

### File Hashes

Use these SHA256 hashes to detect accidental file changes.

```text
1BE5B04A036857AD71E480C772D2CDA0FC1C0850D6CD9AA845B67F3EBEFD8DA5
  D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\displacement_55mesh_300.npy

59E3ED95F7069A798638332238BD780C13F435F6F9DFC17391AAE32D965CACC2
  E:\phd\Code\DM-FEM2D\data\Experiment_300_60\exp_300.txt

8F386D41095B9992949C9EBB939E9F1E6262FC460EC9522A2E241B1149505E07
  E:\phd\Code\DM-FEM2D\data\Experiment_300_60\fu_sim300.txt

D2414083E634B958139C5A4203BFD2C7AFA1782D34D4A80F0F12E669BD8EEEC9
  E:\phd\Code\DM-FEM2D\HydrodynamicData\Yoga\DM10_300_direction0.nc

FDB09EB5149417A0EE3BAB01827F128EF6F1D2A82A0A56709A65422E8A45009B
  E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_MASS1.mtx

4D7B48381323F35210A38469A4F8BC81533FFC57473682BF2108E2A69C5566AA
  E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_STIF1.mtx
```

### Other Related Experimental Files

The same `Experiment_300_60` folder also contains:

```text
exp_60.txt
exp_120.txt
exp_180.txt
exp_240.txt
fu_sim60.txt
fu_sim120.txt
fu_sim180.txt
fu_sim240.txt
fig9_bm_fu.txt
```

These can become additional validation cases after the first baseline is stable.

## Recommended Next Step

Create a tiny, read-only verification script that loads:

- `displacement_55mesh_300.npy`;
- `exp_300.txt`;
- `fu_sim300.txt`;

and recomputes the heave summary and RMSE values above.

This should be the first automated baseline check. It does not require rerunning the full hydroelastic simulation and therefore avoids touching numerical algorithms.

