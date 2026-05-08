# MacBook 代码结构梳理与铰接验证准备

日期：2026-04-29

## 1. 本机定位

MacBook 上的目标目录为：

```text
/Users/yongkang/Library/CloudStorage/OneDrive-宁波东方理工大学/Code_RODM/RODM_20250310
```

未发现名为 `RODM_250310` 的目录；当前按 `RODM_20250310` 继续梳理。该目录与“250310”命名含义一致，应是 2025-03-10 版本。

## 2. 当前结构判断

仓库目前是“原始研究代码 + 标准包骨架 + 验证脚本”的混合状态：

- 根目录保留 `DM_*.py`、`SEREP.py`、`RODM_*.ipynb`，它们仍是论文复现和历史结果溯源基准。
- `src/offshore_energy_sim/` 已经按一体化平台方向拆成 `core`、`geometry`、`environment`、`hydrodynamics`、`structure`、`reduction`、`solver`、`loads`、`response`、`strength`、`power`、`optimization`、`postprocess`、`utils`。
- `scripts/` 中已有轻量核函数验证、配置驱动算例、300 m 参考算例、规则波批量验证和铰接验证入口。
- `docs/` 中已有代码地图、平台结构、Mac 环境、参考算例、回归和铰接报告。

当前最有价值的主线不是继续大搬家，而是先把“可复现验证链路”稳定下来，然后再逐步把旧脚本接入标准包。

## 3. 已做的 Mac 迁移小修正

本次只修改路径解析和文档，不修改 SEREP、频域求解、结构矩阵装配或铰接刚度公式。数值结果预期不变。

- `scripts/run_yoon_hinge_response_validation.py`：仓库根目录改为从脚本位置自动推断；外部数据根目录支持 `RODM_DM_FEM_ROOT`。
- `scripts/run_regular_wave_batch_validation.py`：仓库根目录改为从脚本位置自动推断；外部数据根目录支持 `RODM_DM_FEM_ROOT`。
- `scripts/validate_hinge_model.py`：外部 `DM-FEM2D` 数据根目录支持 `RODM_DM_FEM_ROOT`。
- `scripts/run_hinge_abaqus_benchmark.py`：外部 `DM-FEM2D` 数据根目录支持 `RODM_DM_FEM_ROOT`。
- `docs/macbook_setup.md`：补充 Mac 外部数据路径环境变量说明。
- `src/offshore_energy_sim/README.md`：更新为当前包结构说明，替换过期的“空骨架”描述。

## 4. 外部数据现状

本机 OneDrive 中目前能找到：

- `results/hinge_validation/abaqus_work/Job-1_largemesh_hinge_1.inp`
- `results/hinge_validation/abaqus_work/Job-1_largemesh_hinge_1.dat`
- 历史 Yoon PDF：`A_Work_done/RODM_AD/Hige/Yoon-1-hige-180.pdf`

本次搜索未在 OneDrive 中找到以下关键外部输入：

- `JobMesh5_5_MASS1.mtx`
- `JobMesh5_5_STIF1.mtx`
- `ELEMENTSTIFFNESS_793.mtx`
- `DM10_300_direction0.nc`
- `BM10_145_direaction180.nc`
- `Job-1_largemesh_STIF1.mtx`
- `Job-1_largemesh_ConsistentMass_MASS1.mtx`
- `exp_300.txt`
- `fu_sim300.txt`

因此，当前 Mac 上可以检查代码结构和轻量核函数；完整 300 m RODM 重算、规则波批量验证、严格 Yoon 响应级重算、63 节点铰接矩阵模态复算，都需要先恢复外部 `DM-FEM2D` 数据目录。

## 5. 铰接验证状态

已有报告给出的判断是分层的：

- 铰接矩阵装配内核：`DM_Hinge.py` 与 `offshore_energy_sim.structure.hinges` 的装配核最大误差为 0。
- 63 节点 Abaqus 模态验证：相对于可复现的 `_1.inp` 重运行 dat，20 阶模态最大相对误差约 `3.86e-05`。
- 历史 `Job-1_largemesh_hinge.dat` 与 `_1.inp` 重运行 dat 不一致，不应作为当前 `_1` 算例的通过/失败标准。
- Yoon 响应级验证：当前报告中的 793 节点代理模型可运行，但严格 Yoon 专用模型缺少专用质量矩阵、刚度矩阵和水动力 NetCDF，因此还不能判定为严格完成。

对后续铰接优化来说，建议将“铰接程序准确”拆成三道门槛：

- 单元级门槛：铰接线节点配对、释放 DOF、`+KC/-KC` 块装配必须与 legacy 完全一致。
- 结构级门槛：63 节点铰接模态继续以 `_1.inp` 可复现 dat 为基准，保留最大相对误差阈值。
- 响应级门槛：恢复 Yoon 专用输入后，重新生成单铰/双铰响应曲线、RMSE 和图件，再作为优化前的验收基准。

## 6. 后续工作清单

建议按以下顺序推进：

1. 本机基础 Mac 环境已完成；如需 notebook 或旧可视化脚本，再按需安装 `jupyterlab`、`vtk`。
2. 恢复外部数据：把 Windows 的 `E:\phd\Code\DM-FEM2D` 迁移到 Mac，并设置 `RODM_DM_FEM_ROOT`。
3. 先跑轻量验证：`validate_reduction_solver_kernels.py`、`validate_structure_connectors.py`、`validate_environment_load_power_strength.py`。
4. 再跑参考算例：`verify_reference_case_300.py`、`run_reference_case_300_workflow.py`、`run_regular_wave_batch_validation.py`。
5. 铰接专项：先跑 `validate_hinge_model.py`，再恢复 Yoon 专用输入并跑 `run_yoon_hinge_response_validation.py`。
6. 平台化建模：给 `RodmFrequencyCase` 增加结构模型选择，例如 `rigid`、`hinged`、`spring`，并把 `HingeLineSpec` 纳入配置文件。
7. 优化前准备：建立 `configs/hinge_63_modal_case.yaml`、`configs/yoon_single_hinge_response_case.yaml`、`configs/yoon_double_hinge_response_case.yaml`，把铰接刚度、释放 DOF、铰接位置变成可扫描参数。

## 7. 当前风险

- 旧报告中有 Windows 绝对路径和旧机器 conda 命令，Mac 上不能直接复用，已经开始迁移但还没覆盖所有文档。
- 一些脚本仍依赖外部大文件，不能只凭仓库内 `.npy` 结果证明完整求解链路可复现。
- `optimization/` 目前还是空目录；铰接优化前需要先完成参数化案例对象和验收基准。
- 严格 Yoon 响应级验证的关键输入缺失，这是铰接优化前最重要的阻塞项。

## 8. 2026-04-29 Mac 实测结果

已完成本机 Miniconda 安装：

- Conda 路径：`/Users/yongkang/miniconda3`
- 环境名：`offshore-energy-sim`
- Conda channels：仅使用 `conda-forge`，`channel_priority: strict`
- 核心环境检查：通过
- 可选模块：`vtk`、`jupyterlab` 未安装，暂不影响数值核函数验证

已通过的检查：

- `python -m compileall src scripts`
- `python scripts/check_environment.py`
- `python scripts/validate_reduction_solver_kernels.py`
- `python scripts/validate_structure_connectors.py`
- `python scripts/validate_environment_load_power_strength.py`
- `python scripts/validate_rodm_case_orchestration.py`
- 铰接核函数等价性单独检查：`max_abs_error = 0.0`，`l2_error = 0.0`

当前被外部数据阻塞的检查：

- `python scripts/validate_hinge_model.py`：缺少 `Fem_inp/Job-1_largemesh_hinge_1.inp`、63 节点质量/刚度矩阵等外部输入
- `python scripts/run_yoon_hinge_response_validation.py`：缺少 `StructureData/JobMesh5_5_MASS1.mtx` 等 793 节点结构矩阵、水动力 NetCDF 和 Yoon 专用输入

结论：Mac 环境与包化核心代码已经可用；铰接程序的装配核函数准确性在本机复核通过。下一步不是继续调环境，而是恢复 `DM-FEM2D` 外部数据目录，然后重跑 63 节点模态和 Yoon 响应级验证。
