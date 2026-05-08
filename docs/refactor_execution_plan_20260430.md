# 重构后架构审查与今日最小执行计划

日期：2026-04-30

## 1. 今日审查范围

本次只做架构审查、文件职责梳理、测试方案设计和后续最小修改计划，不新增物理功能，不迁移文件，不修改数值算法。

当前工作区：

```text
/Users/yongkang/Projects/RODM_20250310_local
```

## 2. 当前文件现状

根目录仍承担了较多历史区职责：

| 类型 | 数量 | 当前判断 |
| --- | ---: | --- |
| 根目录 `.py` | 13 | 多数是历史算法脚本或旧入口，短期保留作数值溯源。 |
| 根目录 `.ipynb` | 34 | 应归为论文实验、可视化、历史探索，不应作为后续软件主入口。 |
| 根目录 `.npy` | 6 | 多数是历史基准响应或对比结果，短期保留但需要登记来源。 |
| 根目录 `.inp` | 4 | Abaqus 边界/输入相关产物，短期保留并登记。 |
| `src/offshore_energy_sim/*.py` | 51 | 当前核心方法层，后续新代码应优先进入这里。 |
| `scripts/*.py` | 28 | 当前命令行入口、验证、报告生成和诊断脚本集中区。 |
| `configs/*` | 3 | 已有 RODM 频域算例配置雏形，但覆盖范围还不完整。 |
| `tests/` | 0 | 尚未建立正式最小测试目录。 |
| `notebooks/` | 0 | 尚未建立 notebook 归档目录。 |

## 3. 目录职责建议

建议保持现有科研逻辑不变，先明确边界：

| 目录 | 建议职责 |
| --- | --- |
| `src/offshore_energy_sim/` | 核心可复用方法，包括结构、水动力、降阶、求解、载荷、响应、强度、功率、优化描述对象。 |
| `scripts/` | 薄命令行入口，负责解析参数、调用 `src`、写结果和报告。后续避免在脚本中继续堆复杂算法。 |
| `configs/` | 可复现实验配置，包括结构路径、DOF 约定、主节点、海况、铰接参数、输出路径。 |
| `tests/` | 最小单元测试和 sanity check，优先由现有 `scripts/validate_*` 迁移。 |
| `notebooks/` | 论文图、探索分析、可视化和教学展示。notebook 不作为权威算法入口。 |
| `legacy/` 或 `archive/legacy_scripts/` | 旧 `DM_*.py`、`SEREP.py`、`RODM_*.py` 溯源脚本。迁移前不移动，先登记。 |
| `results/` | 可再生成结果、图件、指标和报告。 |
| `references/` | 外部论文图件、历史程序、数字化曲线和不可轻易覆盖的数据。 |

今日不建议实际移动文件。建议先完成清单和映射表，等 sanity check 建立后再分批迁移。

## 4. 模块划分审查结论

当前 `src/offshore_energy_sim/` 的分层总体合理：

- `core`：案例对象、配置读取、工作流路径。
- `hydrodynamics`：Capytaine NetCDF 和频域水动力项整理。
- `structure`：矩阵读取、装配、连接件、铰接、模块网格。
- `reduction`：DOF 删除、主从 DOF、SEREP。
- `solver`：频域 MCK 求解和 RODM 编排。
- `loads`：载荷映射和风载工具。
- `response`：响应重构和谱后处理。
- `strength`：内力和接口力工具。
- `power`：PV 功率/损失简化模型。
- `validation`：Yoon 铰接和 10x10 模块验证工作流。
- `optimization`：连接件优化变量和目标描述对象。

主要边界问题：

1. `validation/yoon_hinge.py` 和 `validation/complex_hinge_10x10.py` 中仍包含较多求解细节。短期可以接受，因为它们是论文验证工作流；后续若多个案例复用，应逐步把公共结构装配、静态凝聚、水动力整理下沉到 `structure/reduction/hydrodynamics/solver`。
2. `scripts/run_regular_wave_batch_validation.py`、`scripts/run_yoon_hinge_response_validation.py` 仍偏重。后续应保持脚本薄化，但不要在今天改。
3. 海上风机扩展目前只有风载工具，还没有风机气动、塔架、控制或 OpenFAST 类耦合接口。
4. 海上光伏扩展已有 `power/pv.py` 雏形，但还没有阵列遮挡、组件排布、电气损耗、逆变器或 MPPT 模型。

## 5. 科研可靠性审查

已经做得比较好的点：

- 重要公式已开始函数化：DOF 删除、SEREP、MCK 频域求解、JONSWAP、风谱、铰接 `+KC/-KC`、PV 简化功率损失。
- 验证脚本覆盖了核心小函数、结构连接、环境/载荷/功率/强度辅助函数、10x10 节点配对。
- `results/` 和 `docs/` 已保存响应、图件、指标和报告，有利于结果追溯。

仍需补强的点：

- 单位说明不够统一。建议后续在 dataclass 字段名和 docstring 中明确 `_m`、`_rad`、`_deg`、`_n_per_m`、`_kg`、`_n_m` 等。
- 参数硬编码仍较多。短期应先登记，不直接改。
- 旧 notebook 中可能仍有权威历史路径或未迁移算法片段，迁移前需要逐个标注。
- 300 m 历史基准存在水动力节点反序溯源问题，不能静默改默认行为。

## 6. 硬编码参数清单

优先登记以下参数，后续逐步配置化：

| 参数类型 | 当前例子 | 建议归属 |
| --- | --- | --- |
| 结构规模 | `793`、`3965`、`4900`、`100` | `geometry` 或 case config |
| DOF 约定 | 删除第 6 自由度 `[5]`、保留 5DOF | `geometry/reduction` config |
| 主节点规则 | `first_node=424`、`node_interval=6`、`count=10` | `master_node_rule` config |
| 水动力顺序 | `reverse_hydrodynamic_node_order`、`reverse_force_node_order` | `hydrodynamics` config |
| 铰接刚度 | `1e10`、`1e16`、释放刚度 `0/1/10/100` | `structure/hinges` config |
| 静水修正 | `hydrostatic_divisor=1.02/1.05` | `solver` 或 validation case config |
| 频率选择 | `frequency_index=0`、`omega=0.5851` | `solver/environment` config |
| 路径 | `E:\phd\Code\DM-FEM2D`、`/Users/yongkang/data/DM-FEM2D` | `RODM_DM_FEM_ROOT` + config |
| reshape 假设 | `(199, 793)`、`(70, 70)`、`(61, 61)` | response/geometry config |

## 7. 最小测试计划

不建议一开始引入复杂测试框架。可以先用 `pytest` 承接现有验证逻辑，建立最小保护网。

优先测试文件建议：

```text
tests/test_reduction_solver.py
tests/test_structure_connectors.py
tests/test_environment_load_power_strength.py
tests/test_complex_hinge_setup.py
tests/test_reference_sanity.py
```

优先测试函数：

| 模块 | 函数/行为 |
| --- | --- |
| `reduction.dofs` | `retained_dof_indices`、`reduce_matrix_dofs`、`reduce_force_dofs`、`separate_master_slave_dofs`、`reorder_displacement_to_natural_order` |
| `reduction.modal` | `transform_mass_matrix`、小矩阵下的 `serep_reduce` |
| `solver.frequency_domain` | `dynamic_stiffness_matrix`、`solve_frequency_domain` 残差 |
| `structure.assembly` | 节点 DOF 索引、局部矩阵装配 |
| `structure.hinges` | `hinge_coupling_matrix`、`add_hinge_connections_in_place`、`assemble_explicit_hinges_sparse` |
| `structure.modular_grid` | 10x10 主节点、x/y 铰接线、节点配对数量 |
| `hydrodynamics.frequency` | 水动力节点块反序矩阵/力向量 |
| `loads.wind` | 风谱、风阻尼、风力向量 DOF 插入 |
| `power.pv` | 倾角损失、零参考功率除法保护 |

## 8. 论文结果 sanity check

建议建立不依赖外部大文件的只读 sanity check，保护已保存结果：

1. `displacement_55mesh_300.npy`
   - shape = `(3965, 1)`
   - dtype = `complex128`
   - centerline heave length = `60`
   - heave min/max/mean/l2 与已记录基准一致。
2. `results/reference_case_300_rodm_generated.npy`
   - 与默认 packaged solver 历史输出 shape 一致。
   - 与 `DM_Method` 对比应为零误差，若外部数据齐全时运行。
3. `results/reference_case_300_rodm_hydro_reversed.npy`
   - 与 saved baseline 的中心线 heave RMSE 约 `0.0010529883`。
4. `results/regular_wave_batch/wavelength_*m/response.npy`
   - 五个波长响应文件存在，shape 合理。
5. `results/yoon_hinge_standard/*/response.npy`
   - 单铰/双铰响应存在，heave grid shape 与模块布局一致。
6. `results/complex_hinge_10x10/heave_grid_merged.npy`
   - shape = `(61, 61)`。

这些 sanity check 不证明物理模型绝对正确，但能确保重构没有意外改变论文结果。

## 9. 今天建议执行的最小修改顺序

今天建议只做三类低风险工作：

1. 保留本文件作为架构整改清单。
2. 下一步新建 `tests/`，先迁移轻量验证脚本的断言，不改源代码。
3. 新建一份 `docs/root_file_inventory_20260430.md` 或在现有文档中登记根目录 notebook/旧脚本归属。

今天不建议做：

- 不迁移 notebook。
- 不移动根目录 `.npy` 或 `.inp`。
- 不删除旧 `DM_*.py`、`SEREP.py`。
- 不重写 SEREP、频域求解、水动力节点反序、铰接装配。
- 不引入复杂插件式框架或抽象基类。

## 10. 保留不动的锚点

短期必须保留：

- `DM_Method.py`
- `SEREP.py`
- `DM_Assemble.py`
- `DM_Hinge.py`
- `RODM_Wind_main.py`
- 论文复现 notebook
- `displacement_55mesh_300.npy`
- `results/reference_case_300_rodm_generated.npy`
- `results/reference_case_300_rodm_hydro_reversed.npy`
- `results/regular_wave_batch/`
- `results/yoon_hinge_standard/`
- `results/complex_hinge_10x10/`

这些文件是判断“重构是否改变科研结果”的锚点。后续只有在 sanity check 完成后，才建议逐步迁移或归档。
