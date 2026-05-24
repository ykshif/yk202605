# 时域求解器实现状态

日期：2026-05-21

## 1. 已实现内容

已新增标准包：

```text
src/offshore_energy_sim/time_domain/
├── __init__.py
├── cases.py
├── excitation.py
├── hydrodynamic_memory.py
├── postprocess.py
└── solver.py
```

当前能力：

- 固定步长线性 Newmark 时域积分。
- 与现有频域求解器一致的 `exp(-i omega t)` 复幅值相位约定。
- 规则波复激励力到真实时序力的转换。
- 可选余弦 ramp，用于减少启动瞬态。
- 由辐射阻尼矩阵生成 Cummins 辐射记忆 IRF。
- 直接卷积辐射记忆力计算接口。
- 从时序响应反拟合复幅值，并与频域复响应做误差评估。
- 基于现有 `RodmFrequencyCase` 的 RODM 单频时域封装。

当前 RODM 时域封装是第一阶段验证版本：它使用同一个频率点的 `A(omega)` 和 `B(omega)` 作为常系数质量/阻尼项，目标是先证明时间积分、相位约定、响应重构能和现有频域 RODM 对齐。完整 Cummins 直接卷积和状态空间辐射模型已在工具层预留，但尚未接入生产级 RODM case runner。

## 2. 新增验证脚本

```text
scripts/validate_time_domain_rodm_single_frequency.py
```

默认验证 300 m x 60 m、10 个水动力主节点的基础 RODM 算例。脚本流程：

1. 构造或读取 `RodmFrequencyCase`。
2. 运行现有频域求解，得到复幅值基准。
3. 按同一频率生成规则波时序激励。
4. 运行时域 Newmark 积分。
5. 丢弃启动周期后，从时序响应反拟合复幅值。
6. 输出全局 DOF 和主 DOF 的复幅值误差。
7. 绘制中心线 heave 频域/时域拟合对比图。

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_domain_rodm_single_frequency.py
```

如果外部数据不在默认位置，可使用：

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_domain_rodm_single_frequency.py --data-root D:\RODM-data\DM-FEM2D
```

或设置：

```powershell
$env:RODM_DM_FEM_ROOT="D:\RODM-data\DM-FEM2D"
```

通用配置驱动入口也已经支持频域/时域选择：

```powershell
.\.venv\Scripts\python.exe scripts\run_rodm_case_from_config.py --config configs\reference_case_300.yaml --domain frequency
.\.venv\Scripts\python.exe scripts\run_rodm_case_from_config.py --config configs\reference_case_300.yaml --domain time --cycles 80 --steps-per-cycle 180
```

时域结果会写入独立 variant，避免覆盖频域结果：

```text
results/<case_id>/variants/time_domain/response.npy
results/<case_id>/variants/time_domain/time.npy
results/<case_id>/variants/time_domain/master_displacement_time.npy
```

## 3. 当前验证结果

已完成轻量单元验证：

```text
tests/test_time_domain.py
```

覆盖内容：

- 复激励力到真实时序力的相位约定。
- Newmark 时域稳态复幅值与现有频域求解器一致。
- 辐射阻尼余弦变换生成 IRF 的解析函数对照。
- 直接卷积辐射记忆力的历史速度索引。

全量测试：

```text
39 passed
```

本机默认外部数据目录：

```text
C:\Users\WYJ\data\DM-FEM2D
```

当前缺少以下真实 RODM 验证输入，因此脚本只写出了 `missing_inputs` 状态：

```text
C:\Users\WYJ\data\DM-FEM2D\HydrodynamicData\Yoga\DM10_300_direction0.nc
C:\Users\WYJ\data\DM-FEM2D\StructureData\JobMesh5_5_MASS1.mtx
C:\Users\WYJ\data\DM-FEM2D\StructureData\JobMesh5_5_STIF1.mtx
```

输出位置：

```text
results/time_domain/rodm_single_frequency/metrics.json
```

## 4. 数值结果影响

本次新增的是独立时域模块、测试和验证脚本，没有修改现有频域 RODM 求解路径。现有频域结果预期不变。

新增时域结果的当前定位：

- 常系数单频时域模型：用于验证时域积分和相位约定，应与同频频域结果一致。
- IRF 工具：已实现基础变换和直接卷积力计算，但尚未作为默认 RODM 时域生产路径。
- 完整 Cummins/状态空间模型：下一阶段实现。

## 5. 下一步

建议下一步按顺序做：

1. 恢复或指定 `DM-FEM2D` 外部数据目录，运行真实 300 m 单频时域/频域对比。
2. 若误差满足预期，生成宽频 Capytaine 数据集，接入 `radiation_irf_from_damping`。
3. 在 RODM case runner 中增加 `radiation_model=direct_convolution`。
4. 做随机波时域 RMS 与频域谱积分对比。
5. 再实现 WEC-Sim 类似的状态空间辐射近似。
