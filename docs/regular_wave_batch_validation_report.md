# 连续性浮体规则波水弹性对比验证图件

生成时间：2026-04-30 09:30:33

## 1. 验证范围

本报告整理 300 m x 60 m 连续性浮体在规则波波长 60 m、120 m、180 m、240 m、300 m 下的频域水弹性响应图件。
报告按用户要求只保留图片结果和文件索引，不输出 RMSE 或其他误差指标。

## 2. 本机运行策略

- 本机工作目录：`/Users/yongkang/Projects/RODM_20250310_local`
- 外部数据根目录：`/Users/yongkang/data/DM-FEM2D`
- 若外部矩阵、水动力和对比曲线齐全，脚本会重新计算并绘制对比图。
- 若外部大文件尚未迁移到 Mac，脚本会复用本机已有 `response.npy` 和历史对比图，避免覆盖已有验证图件。
- 300 m 波长按既有溯源结论使用 `hydro_reversed` 响应；60-240 m 仍使用默认水动力节点顺序。

## 3. 汇总图

- `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/figures/regular_wave_60_300m_comparison_panel.png`

## 4. 分波长图件

| 波长 (m) | 响应来源 | 图件状态 | PNG 图件 |
| ---: | --- | --- | --- |
| 60 | 复用已有响应 | 沿用已有对比图 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_60m/figures/regular_wave_60m_heave_comparison.png` |
| 120 | 复用已有响应 | 沿用已有对比图 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_120m/figures/regular_wave_120m_heave_comparison.png` |
| 180 | 复用已有响应 | 沿用已有对比图 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_180m/figures/regular_wave_180m_heave_comparison.png` |
| 240 | 复用已有响应 | 沿用已有对比图 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_240m/figures/regular_wave_240m_heave_comparison.png` |
| 300 | 复用已有 hydro_reversed 响应 | 绘制300m方向修正图 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_300m/figures/regular_wave_300m_heave_selected.png` |

## 5. 方向约定说明

既有溯源报告显示，300 m 历史保存基准 `displacement_55mesh_300.npy` 更接近水动力节点反序候选结果；因此本图件流程将 300 m 标记为 `reverse_hydrodynamic_node_order = true`。
这不是横坐标简单反画，而是 10 个水动力节点块与结构主节点排列之间的顺序约定差异。

## 6. 数据状态

波长 60 m 缺少以下外部输入：
- `/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_60_direction0.nc`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/exp_60.txt`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/fu_sim60.txt`

波长 120 m 缺少以下外部输入：
- `/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_120_direction0.nc`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/exp_120.txt`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/fu_sim120.txt`

波长 180 m 缺少以下外部输入：
- `/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_180_direction0.nc`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/exp_180.txt`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/fu_sim180.txt`

波长 240 m 缺少以下外部输入：
- `/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_240_direction0.nc`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/exp_240.txt`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/fu_sim240.txt`

波长 300 m 缺少以下外部输入：
- `/Users/yongkang/data/DM-FEM2D/HydrodynamicData/Yoga/DM10_300_direction0.nc`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx`
- `/Users/yongkang/data/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/exp_300.txt`
- `/Users/yongkang/data/DM-FEM2D/data/Experiment_300_60/fu_sim300.txt`

## 7. 结论

本机副本中已经具备五个波长的响应文件和对比图件，可用于连续性浮体规则波水弹性结果查看。
若需要完全重新求解，应先迁移 `DM-FEM2D` 外部数据目录并设置 `RODM_DM_FEM_ROOT`，再重新运行本脚本。
