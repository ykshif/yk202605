# 一体化平台重构状态说明

生成时间：2026-04-28

## 1. 当前定位

当前仓库已经从原始论文复现脚本，逐步拆分出 `src/offshore_energy_sim` 标准包。原始 `DM_*.py`、`RODM_*.ipynb` 和数据文件继续保留，作为论文复现与结果溯源基准；新增代码只承担封装、验证、批量运行和后续平台化入口职责。

本轮没有修改原始数值算法。预期数值结果不应因接口整理发生变化。

## 2. 已形成的标准模块

| 目录 | 当前职责 |
| --- | --- |
| `core` | 路径、配置、案例对象、工作流输出路径 |
| `geometry` | 浮体几何参数对象 |
| `environment` | 波浪参数与谱模型 |
| `hydrodynamics` | NetCDF 水动力数据读取和频率数据处理 |
| `structure` | Abaqus 矩阵读取、矩阵装配、连接器、铰接、RODM 结构降阶准备 |
| `reduction` | 自由度选择与模态工具 |
| `solver` | 频域求解和 RODM 频域案例封装 |
| `loads` | 荷载映射与风荷载入口 |
| `response` | 响应谱、保留自由度响应、重构入口 |
| `strength` | 内力/强度后处理雏形 |
| `power` | PV 发电损失/功率模型入口 |
| `postprocess` | 指标、绘图、验证报告生成 |
| `utils` | 哈希等通用工具 |

## 3. 本轮新增铰接标准包

铰接入口位于 `src/offshore_energy_sim/structure/hinges.py`：

- `HingeLineSpec`：描述铰接线两侧节点列、网格尺寸、铰接刚度和释放自由度；
- `apply_hinge_line_in_place`：向全局刚度矩阵添加铰接连接；
- `remove_hinge_line_elements_in_place`：可选移除铰接线两侧壳单元刚度；
- `build_hinged_stiffness`：组合生成铰接刚度矩阵，供后续优化和参数扫描调用。

验证脚本：

- `scripts/validate_structure_connectors.py`：验证结构装配、连接器和铰接核函数；
- `scripts/run_hinge_abaqus_benchmark.py`：在 `results/hinge_validation/abaqus_work` 中重运行 63 节点 Abaqus 铰接算例；
- `scripts/validate_hinge_model.py`：生成铰接程序包验证报告。

验证结论见 `docs/hinge_model_validation_report.md`。新铰接接口与旧 `DM_Hinge.py` 装配核函数最大误差为 0；63 节点矩阵与 `_1.inp` 重运行 Abaqus dat 的 20 阶模态最大相对误差约 `3.86e-05`。

## 4. 已移除的常规分支

水动力节点反序对比不再作为常规验证路径。相关历史结果只作为数据溯源参考，不进入主回归套件和规则波批量验证主线。

当前规则波验证脚本 `scripts/run_regular_wave_batch_validation.py` 只输出默认 RODM / `DM_Method` 等价路径与实验、Fu 仿真的对比。

## 5. 下一步建议

1. 将铰接模型接入 RODM 频域案例对象，使 `RodmFrequencyCase` 可选择 `structure_model=rigid/hinged/spring`。
2. 将 63 节点铰接算例扩展为规则波频域响应验证，而不仅是结构模态验证。
3. 整理 `DM_Hinge_test.ipynb` 和 `RODM_Hige_study*.ipynb` 中仍未固化的参数、路径和图表。
4. 建立 `configs/hinge_63_modal_case.yaml` 和后续 `configs/hinge_63_frequency_case.yaml`。
5. 再处理 SEREP/响应重构模块，将铰接响应也纳入统一后处理和绘图接口。
