# RODM 标准化代码用户说明书

日期：2026-04-30

这份说明书面向后续使用人员，目标是用最短路径理解：代码在哪里、怎么运行、每个包做什么、每个核心函数负责什么。如果只是复现实验和论文验证，优先使用 `scripts/` 中的入口脚本，不建议直接运行旧 notebook。

如果需要逐一查看每个文件夹和每个 `.py` 文件的职责，请阅读更完整的说明：

```text
docs/full_code_user_manual_cn.md
```

## 1. 一句话理解当前代码

当前代码已经从“研究 notebook”整理成“标准 Python 包 + 可重复运行脚本 + 中文报告”的结构。旧 notebook 保留为数值溯源，新代码入口集中在：

```text
/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim
/Users/yongkang/Projects/RODM_20250310_local/scripts
/Users/yongkang/Projects/RODM_20250310_local/docs
```

## 2. 最常用命令

检查环境：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/check_environment.py
```

运行 Yoon 单铰/双铰验证：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --case all
```

生成连续性浮体与铰接浮体统一验证报告：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/build_hydroelastic_validation_report.py
```

打开水动力 `.nc` 计算窗口：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_hydrodynamics_ui.py --host 127.0.0.1 --port 8765
```

窗口用户说明：

```text
docs/hydrodynamics_ui_user_guide_cn.md
```

只运行 10x10 输入检查，不求解：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_complex_hinge_10x10.py --skip-solve
```

10x10 结构矩阵到位后运行完整水弹性计算：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_complex_hinge_10x10.py
```

检查 10x10 铰接节点生成规则：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/validate_complex_hinge_10x10_setup.py
```

## 3. 数据放在哪里

默认外部数据根目录：

```text
/Users/yongkang/data/DM-FEM2D
```

当前常用数据结构：

| 数据类型 | 默认目录 | 说明 |
| --- | --- | --- |
| Yoon 铰接结构矩阵 | `StructureData/Yoon_hinge` | 单铰、双铰、斜入射双铰使用。 |
| Yoon/10x10 水动力数据 | `HydrodynamicData/Yoon_hinge` | 包含 `DM10_direction*.nc` 和 `DM10_10_direction0_wl180.nc`。 |
| 10x10 结构矩阵 | `StructureData/Hinge_complex_paper4` | 需要 `Job3030hinge-1_MASS1.mtx` 和 `Job3030hinge-1_STIF1.mtx`。 |
| 论文参考数据 | `references/hinge_published` | 数字化 CSV、历史论文 PDF 图件、旧程序副本。 |

如果数据放在其他地方，可以使用：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --data-root /path/to/DM-FEM2D
```

或者设置环境变量：

```bash
export RODM_DM_FEM_ROOT=/path/to/DM-FEM2D
```

## 4. 代码结构速览

| 包 | 给使用者的解释 | 典型文件 |
| --- | --- | --- |
| `core` | 算例配置、JSON 报告、工作流工具。 | `workflow.py`、`config.py` |
| `hydrodynamics` | 读取 Capytaine NetCDF，整理附加质量、阻尼、静水刚度和波浪力。 | `netcdf.py`、`frequency.py` |
| `structure` | 读取 Abaqus 矩阵，生成模块网格，装配铰接和连接件刚度。 | `matrix_io.py`、`hinges.py`、`modular_grid.py` |
| `reduction` | 删除自由度、划分主从 DOF、SEREP、静态凝聚和响应重排。 | `dofs.py`、`modal.py` |
| `solver` | 建立并求解频域动力方程。 | `frequency_domain.py` |
| `response` | 把降阶响应恢复到全局结构节点。 | `reconstruction.py` |
| `validation` | 标准验证算例，包括 Yoon 单铰/双铰和 10x10 铰接。 | `yoon_hinge.py`、`complex_hinge_10x10.py` |
| `optimization` | 后续连接件位置和刚度优化的变量、目标函数、问题描述。 | `connectors.py` |
| `postprocess` | 绘图、报告、对比结果后处理。 | `plots.py`、`validation.py` |

## 5. 入口脚本说明

| 脚本 | 什么时候用 | 输出 |
| --- | --- | --- |
| `scripts/run_yoon_hinge_cases.py` | 复现单铰、双铰和斜入射双铰结果。 | `results/yoon_hinge_standard` 下的响应、图片和报告。 |
| `scripts/run_complex_hinge_10x10.py` | 检查或运行 10x10 模块铰接水弹性算例。 | `results/complex_hinge_10x10` 下的响应、图片和报告。 |
| `scripts/run_hydrodynamics_ui.py` | 打开本地水动力计算窗口，输入浮体阵列参数并生成 Capytaine `.nc` 文件。 | `results/hydrodynamics_ui` 下的水动力 NetCDF 文件。 |
| `scripts/validate_hydrodynamics_ui_against_nc.py` | 通过 UI API 重新生成小型水动力文件，并和已有 Yoon/10x10 `.nc` 做对比验证。 | `results/hydrodynamics_ui_validation` 下的 `.nc`、JSON 和 Markdown 报告。 |
| `scripts/build_hydroelastic_validation_report.py` | 汇总连续性浮体 60-300 m 与单铰/双铰浮体对比结果。 | `docs/hydroelastic_validation_report.md` 和 `results/hydroelastic_validation`。 |
| `scripts/validate_complex_hinge_10x10_setup.py` | 检查 10x10 节点编号和铰接线数量是否正确。 | 终端打印验证通过信息。 |
| `scripts/validate_structure_connectors.py` | 检查结构连接、铰接装配核函数是否正常。 | 终端打印每项检查结果。 |
| `scripts/validate_published_hinge_kernels.py` | 检查已发表论文旧程序和新铰接核函数是否等价。 | `results/hinge_published_validation` 和审计文档。 |

## 6. 核心函数功能索引

### 6.1 Yoon 单铰/双铰

| 函数 | 功能 |
| --- | --- |
| `build_yoon_hinge_cases()` | 构造全部标准 Yoon 铰接验证算例。 |
| `missing_input_paths(case)` | 检查算例所需结构矩阵和水动力文件是否缺失。 |
| `solve_yoon_hinge_case(case)` | 完成单个 Yoon 算例的结构装配、降阶、水动力耦合和频域求解。 |
| `extract_yoon_hinge_heave_grid(case, response)` | 从全局响应中抽取 heave 并整理成论文对比网格。 |
| `plot_yoon_hinge_case(result, output_dir)` | 输出 Yoon 对比图。 |

### 6.2 10x10 模块铰接

| 函数 | 功能 |
| --- | --- |
| `build_complex_hinge_10x10_case()` | 构造 10x10 标准算例，包括 100 个模块、180 条铰接线和 100 个主节点。 |
| `solve_complex_hinge_case(case)` | 完成 10x10 稀疏结构装配、静态凝聚、水动力耦合和频域求解。 |
| `extract_complex_hinge_heave_grid(case, response)` | 把 4900 个节点的 heave 响应整理成 70x70 或 61x61 网格。 |
| `plot_complex_hinge_result(result, output_dir)` | 输出 10x10 heave 热力图和中心线图。 |

### 6.3 模块网格与铰接

| 函数 | 功能 |
| --- | --- |
| `generate_master_nodes_one_based(grid)` | 返回每个模块中心节点编号。 |
| `generate_x_hinge_node_pairs(grid)` | 生成左右相邻模块之间的边界节点对。 |
| `generate_y_hinge_node_pairs(grid)` | 生成上下相邻模块之间的边界节点对。 |
| `generate_grid_hinge_specs(grid)` | 生成全部 x/y 铰接线，并设置不同释放 DOF。 |
| `drop_duplicate_module_interfaces(...)` | 删除模块界面重复节点行列，得到连续响应场。 |
| `hinge_coupling_matrix(...)` | 生成 `KC` 铰接耦合矩阵。 |
| `assemble_explicit_hinges_sparse(...)` | 稀疏装配大规模铰接刚度，服务 10x10 算例。 |

### 6.4 优化预留

| 函数或类 | 功能 |
| --- | --- |
| `ConnectorDesignVariable` | 描述一个连接件设计变量，例如刚度或释放刚度。 |
| `ConnectorObjectiveSpec` | 描述优化目标，例如最大 heave 或连接件力。 |
| `ConnectorOptimizationProblem` | 把算例、变量、目标和约束组合成优化问题。 |
| `uniform_hinge_stiffness_variables()` | 快速生成 x/y 两个统一铰接刚度变量。 |

## 7. 新增一个算例应该怎么做

1. 在 `validation` 中新增一个 case builder，例如 `build_xxx_case()`。
2. 明确结构矩阵路径、水动力路径、节点数、主节点、删除 DOF、降阶方法。
3. 如果是铰接模型，用 `ExplicitHingeSpec` 或 `generate_grid_hinge_specs()` 定义连接件。
4. 写一个 `solve_xxx_case()`，复用 `reduction`、`solver` 和 `response` 中已有函数。
5. 写一个 `scripts/run_xxx.py`，负责命令行参数、输入检查、保存结果和写报告。
6. 增加一个轻量验证脚本，至少检查节点编号、矩阵维度和输出形状。

## 8. 后续做连接件优化怎么接

建议分三步：

1. 全局刚度优化：先优化两个变量 `k_hinge_x` 和 `k_hinge_y`，确认优化流程能跑通。
2. 分区刚度优化：把连接件分为边界、角部、中心区域，优化多个区域刚度。
3. 位置与拓扑优化：再研究连接线是否启用、连接件位置移动、局部释放自由度组合等离散变量。

优化目标可以从简单到复杂：

| 目标 | 说明 |
| --- | --- |
| `max_heave` | 最小化最大 heave 响应。 |
| `mean_heave` | 最小化平均 heave 响应。 |
| `max_connector_force` | 控制连接件最大受力。 |
| `mean_connector_force` | 控制整体连接件平均受力。 |
| `custom` | 预留给局部变形、频带平均、强度或功率损失目标。 |

## 9. 当前重要状态

Yoon 单铰/双铰验证链路已经完成，可以作为铰接程序 baseline。10x10 标准程序已经完成结构设计和输入检查，当前还等待两个结构矩阵文件同步：

```text
/Users/yongkang/data/DM-FEM2D/StructureData/Hinge_complex_paper4/Job3030hinge-1_MASS1.mtx
/Users/yongkang/data/DM-FEM2D/StructureData/Hinge_complex_paper4/Job3030hinge-1_STIF1.mtx
```

这两个文件到位后，直接运行 `scripts/run_complex_hinge_10x10.py` 即可输出 10x10 响应和图片。
