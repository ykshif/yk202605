# 连续性浮体与铰接浮体水弹性统一验证报告

生成时间：2026-04-30 09:39:19

## 1. 报告目标

本报告把当前标准化代码的两条核心验证线合并到同一份说明中：

- 连续性浮体：300 m x 60 m 连续体浮体，规则波波长 60 m、120 m、180 m、240 m、300 m。
- 铰接浮体：单铰接约束和双铰接约束，并与实验结果、他人数值结果或历史论文图件进行对比。

报告侧重可读性和图件索引，不在正文中展开 RMSE 等误差表。数值算法以当前已验证的标准脚本为准。

## 2. 代码入口

| 验证对象 | 标准脚本 | 结果目录 |
| --- | --- | --- |
| 连续性浮体 60-300 m | `/Users/yongkang/Projects/RODM_20250310_local/scripts/run_regular_wave_batch_validation.py` | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch` |
| 单铰/双铰浮体 | `/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py` | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard` |
| 本统一报告 | `/Users/yongkang/Projects/RODM_20250310_local/scripts/build_hydroelastic_validation_report.py` | `/Users/yongkang/Projects/RODM_20250310_local/results/hydroelastic_validation` |

## 3. 连续性浮体规则波验证

连续性浮体计算采用 RODM 频域水弹性流程：读取结构质量/刚度矩阵和 Capytaine 水动力数据，删除每节点第 6 自由度，选取 10 个主节点降阶，求解频域动力方程，并提取中心线 heave RAO 与实验/他人数值结果对比。

### 3.1 五个波长汇总图

![连续性浮体五个波长汇总](/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/figures/regular_wave_60_300m_comparison_panel.png)

### 3.2 分波长结果索引

| 波长 (m) | 响应状态 | 图件状态 | 300 m 方向修正 | 图件 |
| ---: | --- | --- | --- | --- |
| 60 | 复用已有响应 | 沿用已有对比图 | 否 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_60m/figures/regular_wave_60m_heave_comparison.png` |
| 120 | 复用已有响应 | 沿用已有对比图 | 否 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_120m/figures/regular_wave_120m_heave_comparison.png` |
| 180 | 复用已有响应 | 沿用已有对比图 | 否 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_180m/figures/regular_wave_180m_heave_comparison.png` |
| 240 | 复用已有响应 | 沿用已有对比图 | 否 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_240m/figures/regular_wave_240m_heave_comparison.png` |
| 300 | 复用已有 hydro_reversed 响应 | 绘制300m方向修正图 | 是 | `/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/wavelength_300m/figures/regular_wave_300m_heave_selected.png` |

### 3.3 连续体当前状态

当前本机已有五个波长的响应数组和对比图件。若外部 `DM-FEM2D` 大文件不完整，脚本会复用已有响应和历史图件；若数据完整，脚本会自动重新计算。
300 m 波长按照既有溯源结果采用水动力节点反序候选，即 `reverse_hydrodynamic_node_order = true`，这是水动力节点块与结构主节点排列的顺序约定修正，不是简单把横坐标反画。
300 m 偏差专项诊断见：`/Users/yongkang/Projects/RODM_20250310_local/docs/regular_wave_300m_diagnostic_report.md`

连续性浮体单独报告：`/Users/yongkang/Projects/RODM_20250310_local/results/regular_wave_batch/regular_wave_batch_validation_report.md`

## 4. 铰接浮体约束验证

铰接模型使用 `ExplicitHingeSpec` 定义节点对连接。每对铰接节点在全局刚度矩阵中加入 `+KC/-KC` 四个块，未释放自由度使用大刚度约束相对位移，释放转动自由度使用小惩罚刚度保留数值稳定性。

### 4.1 约束设置

| 算例 | 类型 | 模块数 | 铰接线 | 每线节点对 | 释放 DOF | 释放刚度 | 求解状态 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `single_180` | 单铰接 | 2 | 1 | 13 | [4] | 100.0 | solved |
| `double_180` | 双铰接 | 3 | 2 | 13 | [4] | 100.0 | solved |
| `double_210` | 双铰接 | 3 | 2 | 13 | [4] | 100.0 | solved |
| `double_240` | 双铰接 | 3 | 2 | 13 | [4] | 100.0 | solved |
| `double_270` | 双铰接 | 3 | 2 | 13 | [4] | 100.0 | solved |

### 4.2 对比结果索引

| 算例 | 对比对象 | 代表性图件 | 说明 |
| --- | --- | --- | --- |
| `single_180` | 历史论文图 1 张 | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/single_180/comparison_panels/single_180_centerline_current_vs_legacy.png` | 已完成响应计算；未找到数字化参考曲线，已附历史论文图件渲染。 |
| `double_180` | 他人数值曲线 3 条，实验点 1 组，历史论文图 3 张 | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_180/comparison_panels/double_180_case_2_centerline_current_vs_legacy.png` | 已完成响应计算和对比图输出。 |
| `double_210` | 他人数值曲线 3 条，历史论文图 3 张 | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_210/comparison_panels/double_210_case_2_centerline_current_vs_legacy.png` | 已完成响应计算和对比图输出。 |
| `double_240` | 历史论文图 3 张 | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_240/comparison_panels/double_240_case_2_centerline_current_vs_legacy.png` | 已完成响应计算；未找到数字化参考曲线，已附历史论文图件渲染。 |
| `double_270` | 历史论文图 3 张 | `/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_270/comparison_panels/double_270_case_2_centerline_current_vs_legacy.png` | 已完成响应计算；未找到数字化参考曲线，已附历史论文图件渲染。 |

### 4.3 代表性图件

#### 单铰接中心线对比

![单铰接中心线对比](/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/single_180/comparison_panels/single_180_centerline_current_vs_legacy.png)

#### 双铰接 180 度中心线对比

![双铰接 180 度中心线对比](/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_180/comparison_panels/double_180_case_2_centerline_current_vs_legacy.png)

#### 双铰接 210 度中心线对比

![双铰接 210 度中心线对比](/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/double_210/comparison_panels/double_210_case_2_centerline_current_vs_legacy.png)

### 4.4 铰接当前状态

单铰、双铰和斜入射双铰均已完成标准脚本求解。双铰 180 度和 210 度有数字化他人数值曲线，双铰 180 度中心线还包含实验点。单铰接当前没有可靠的单铰数字化 CSV，因此报告采用当前 RODM 结果与历史论文图件的视觉对比，避免误用双铰数据。

铰接单独报告：`/Users/yongkang/Projects/RODM_20250310_local/results/yoon_hinge_standard/report.md`

## 5. 统一结论

- 连续性浮体 60 m、120 m、180 m、240 m、300 m 五个波长已经具备响应文件和图件型对比结果。
- 铰接浮体已经实现单铰接和双铰接约束，并通过标准入口生成对比图件。
- 当前代码已经把连续体水弹性计算、铰接约束装配、结果绘图和报告生成分开，后续可继续扩展到 10x10 模块和连接件刚度/位置优化。

## 6. 复现命令

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_regular_wave_batch_validation.py
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --case all
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/build_hydroelastic_validation_report.py
```
