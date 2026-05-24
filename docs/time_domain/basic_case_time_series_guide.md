# 300 m 基础 RODM 算例时序计算指南

日期：2026-05-21

## 1. 目标

本指南说明如何运行当前第一版时域水弹性计算：

- 几何：300 m x 60 m 连续性浮体。
- 水动力主节点：10 个模块/主节点。
- 结构：793 个 FEM 节点，删除第 6 自由度后保留 5DOF。
- 时域方法：线性 Newmark 积分。
- 当前水动力模型：选定频率点的 `A(omega)` 和 `B(omega)` 常系数模型。
- 验证：从时域稳态响应反拟合复幅值，并与现有频域 RODM 解对比。

这一步的目的不是替代完整 Cummins 记忆模型，而是先完成“基础算例可以生成稳定时间序列，并能和频域稳态结果对齐”。

## 2. 数据准备

脚本默认查找：

```text
C:\Users\WYJ\data\DM-FEM2D
```

需要至少包含：

```text
HydrodynamicData\Yoga\DM10_300_direction0.nc
StructureData\JobMesh5_5_MASS1.mtx
StructureData\JobMesh5_5_STIF1.mtx
```

如果数据在其他目录，可以设置环境变量：

```powershell
$env:RODM_DM_FEM_ROOT="D:\RODM-data\DM-FEM2D"
```

或运行时传入：

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py --data-root D:\RODM-data\DM-FEM2D
```

## 3. 推荐运行命令

基础运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py --data-root D:\RODM-data\DM-FEM2D
```

减少计算量的快速试算：

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py --data-root D:\RODM-data\DM-FEM2D --cycles 20 --steps-per-cycle 90 --skip-frequency-validation
```

用于稳态频域对比的推荐设置：

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py --data-root D:\RODM-data\DM-FEM2D --cycles 80 --steps-per-cycle 180 --ramp-cycles 5 --discard-cycles 55
```

默认使用 300 m 算例中更接近历史基准的水动力节点反序候选：

```text
--hydro-node-order reversed
```

如需使用原始节点顺序：

```powershell
.\.venv\Scripts\python.exe scripts\run_time_domain_reference_case_300.py --data-root D:\RODM-data\DM-FEM2D --hydro-node-order default
```

## 4. 输出位置

默认输出：

```text
results/time_domain/reference_case_300_timeseries/
```

主要文件：

```text
time.npy
global_displacement_time.npy
master_displacement_time.npy
master_velocity_time.npy
master_acceleration_time.npy
centerline_heave_time.npy
centerline_representative_heave.csv
metrics.json
report.md
figures/
```

数组含义：

- `time.npy`：时间序列，shape 为 `(n_time,)`。
- `global_displacement_time.npy`：全局保留 DOF 位移时序，shape 为 `(n_time, 793*5)`。
- `master_displacement_time.npy`：10 个水动力主节点的主自由度位移时序，shape 为 `(n_time, 10*5)`。
- `centerline_heave_time.npy`：中心线 heave 时序，shape 为 `(n_time, 60)`。
- `centerline_representative_heave.csv`：中心线首端、中点、末端三个代表点的 heave 时序。

图件：

- `centerline_representative_heave_time.png`：三个代表位置的 heave 时间历程。
- `centerline_heave_snapshots.png`：后段若干时刻的中心线 heave 快照。
- `centerline_heave_frequency_validation.png`：时域拟合幅值和频域幅值对比。

## 5. 判断结果是否正常

首先看 `metrics.json`：

```text
status: completed
global_amplitude_error.l2_relative_error
master_amplitude_error.l2_relative_error
```

如果时域相位、积分和重构都正确，充分长时间并丢弃 ramp 瞬态后，时域反拟合幅值应接近频域结果。若误差偏大，优先检查：

- `cycles` 是否太少；
- `discard-cycles` 是否小于 ramp 后稳定时间；
- `steps-per-cycle` 是否太低；
- 是否使用了正确的水动力节点顺序；
- 外部数据是否与当前基准一致。

## 6. 当前限制

当前脚本使用单频常系数模型：

```text
(M_R + A(omega_ref)) qdd + B(omega_ref) qd + K q = F_exc(t)
```

完整 Cummins 形式：

```text
(M_R + A_inf) qdd + K q + integral K_rad(t-tau) qd(tau) dtau = F_exc(t)
```

已经在 `src/offshore_energy_sim/time_domain` 中实现了 IRF 和直接卷积基础工具，但尚未作为该基础算例脚本的默认生产路径。下一阶段会把 `radiation_model=direct_convolution` 接入这个脚本。
