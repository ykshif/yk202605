# Local Data Layout for the 300 m RODM Time-Domain Case

The current local mirror for the basic 300 m, 10-module case is:

```text
data/external/DM-FEM2D/
  HydrodynamicData/Yoga/DM10_300_direction0.nc
  HydrodynamicData/Yoga/BM10_direaction0_full.nc
  HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh4.nc
  HydrodynamicData/Yoga/DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc
  StructureData/JobMesh5_5_MASS1.mtx
  StructureData/JobMesh5_5_STIF1.mtx
  data/Experiment_300_60/exp_300.txt
  data/Experiment_300_60/fu_sim300.txt
```

The files were copied from:

```text
E:/phd/Code/DM-FEM2D/HydrodynamicData
E:/phd/Code/DM-FEM2D/StructureData
E:/phd/Code/DM-FEM2D/data/Experiment_300_60
```

The repository `.gitignore` excludes `data/`, so this is a local working-data
mirror rather than version-controlled source code.

Reference hashes for the three required solver inputs:

```text
DM10_300_direction0.nc  D2414083E634B958139C5A4203BFD2C7AFA1782D34D4A80F0F12E669BD8EEEC9
BM10_direaction0_full.nc F8F0AB6D2D555C9B1B9CCCBE204BB541F89D662E4C4AE53E59BC83CC19651590
DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh4.nc D685585645470B420194A6BD77F78AAF31E614D585B4EADB0857AA9EEAC3FB56
DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc C6AF9671D59198CC15D65CBE91160DF4892D432D6EE3AAD25FF17D1F34DCA002
JobMesh5_5_MASS1.mtx   FDB09EB5149417A0EE3BAB01827F128EF6F1D2A82A0A56709A65422E8A45009B
JobMesh5_5_STIF1.mtx   4D7B48381323F35210A38469A4F8BC81533FFC57473682BF2108E2A69C5566AA
```

`DM10_300_direction0.nc` is the correct single-frequency 300 m benchmark file.
It contains one omega value, so it supports the current `radiation_model=constant`
time-domain validation. A Cummins `radiation_model=direct_convolution` run still
needs a compatible multi-frequency 10-module hydrodynamic dataset. The local
`BM10_direaction0_full.nc` file has 40 omega values from 0.1 to 2.0 rad/s and
60 hydrodynamic DOFs, so it is a useful candidate for the next direct-convolution
development step. Its nearest grid point to the 300 m benchmark omega 0.4157
rad/s is 0.3923076923 rad/s, so strict 300 m validation will require either a
matching multi-frequency DM10 file or frequency interpolation.

`DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh4.nc` is the locally
generated DM10 Cummins development dataset. It contains 42 omega values from
0.1 to 2.0 rad/s plus the exact 300 m benchmark omega 0.4157 rad/s. It is a
fast mesh-4 dataset for method validation; production runs should regenerate it
with a finer panel mesh and documented convergence checks.

`DM10_direction0_cummins_omega0p10_2p00_41plus_target_mesh2.nc` is the finer
mesh-2 dataset generated for the next Cummins validation step. It uses the same
frequency grid and geometry as the mesh-4 file and should be the default local
dataset for regular-wave Cummins validation from this point onward.
