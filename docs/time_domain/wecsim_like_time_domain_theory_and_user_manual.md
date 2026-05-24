# RODM WEC-Sim-like 时域平台理论说明与用户手册

更新时间：2026-05-22

本文档说明当前 RODM 外接 WEC-Sim-like 时域平台的理论基础、代码实现逻辑、验证方法和用户使用方式。当前平台已经支持：

```text
1. 基于 RODM 频域水弹性结果的外接时域求解；
2. WEC-Sim/Cummins 思想下的直接卷积辐射记忆力；
3. ERA 状态空间辐射近似；
4. Newmark 平均加速度法；
5. 显式四阶 Runge-Kutta, RK4, 作为交叉验证积分器；
6. 先降维质量/刚度/水动力矩阵，再在 reduced/master DOF 上推进；
7. 需要全局结果时再由 SEREP/T 矩阵还原到 global retained DOF；
8. 规则波和波谱输入；
9. 简单线性系泊 stiffness provider 接口；
10. 多海况、长时间、频域 RMS、时域 RMS、时间序列对比验证。
```

最重要的架构边界保持不变：

```text
RODM 频域水弹性模型仍是主程序和核心方法；
WEC-Sim/Cummins/状态空间方法只作为外接 time-domain adapter；
time_domain_adapter 读取 RODM 已导出的频域水动力、结构矩阵和响应数据；
RODM 频域核心不依赖、不调用、不嵌入时域求解器；
不修改 RODM 的 SEREP、T 矩阵、控制点、模块划分和频域响应求解流程。
```

## 1. 理论部分

### 1.1 总体定位

RODM 负责在频域中求解大型柔性浮体的水弹性问题，输出或提供：

```text
M_struct          结构质量矩阵
K_struct          结构刚度矩阵
A(omega)          频域附加质量
B(omega)          频域辐射阻尼
F_ex(omega)       频域波浪激励力
K_hs              静水恢复项
RAO/response      频域响应
T/SEREP           降维和全局还原关系
```

时域平台不重新定义这些核心对象，而是在 adapter 层把它们转换为线性时域动力方程。

### 1.2 Reduced-Space 动力学方程

当前时域求解全部在 reduced/master DOF 空间进行。设 reduced 位移为 `q(t)`，则线性时域方程为：

```text
[M + A_inf + A_res] qdd(t)
+ [C_res + C_other] qd(t)
+ [K_struct + K_hs + K_moor] q(t)
+ F_rad_memory(t)
= F_exc(t) + F_ext(t)
```

其中：

```text
M                  reduced structural mass
A_inf              infinite-frequency added mass
A_res              residual added-mass correction
C_res              residual radiation damping correction
C_other            future linear damping/PTO/control interface
K_struct           reduced structural stiffness
K_hs               reduced hydrostatic stiffness
K_moor             optional reduced mooring stiffness
F_rad_memory       radiation memory force
F_exc              wave excitation time series
F_ext              future external load interface
```

全局响应不在每一个动力学方程内部推进，而是在 reduced 响应求解完成后再还原：

```text
u_global(t) = T q(t)
```

这样做有两个目的：

```text
1. 降低时域推进规模；
2. 保持 RODM 的结构降维和全局还原逻辑只作为已有频域结果的使用者，不让时域模块反向改变 RODM 主程序。
```

### 1.3 Cummins 方程

WEC-Sim 的线性时域水动力框架基于 Cummins 方程。辐射力分为高频附加质量项和速度历史卷积项：

```text
F_rad(t) = A_inf qdd(t) + integral_0^t K_r(t - tau) qd(tau) d tau
```

把高频附加质量移入左端后：

```text
[M + A_inf] qdd(t)
+ integral_0^t K_r(t - tau) qd(tau) d tau
+ K_total q(t)
= F_exc(t)
```

当前实现为了处理有限频带水动力数据，还支持 residual 修正：

```text
M_eff = M + A_inf + A_res
C_eff = C_res
K_eff = K_struct + K_hs + K_moor
```

`A_res` 和 `C_res` 用于让有限频带 Cummins 核在选定目标频率附近更接近原始频域水动力。

### 1.4 Radiation Kernel

辐射记忆核由频域辐射阻尼生成：

```text
K_r(t) = 2/pi * integral_0^infty B(omega) cos(omega t) d omega
```

数值实现只拥有有限频率范围，因此需要检查：

```text
omega_min, omega_max, delta_omega
frequency grid 是否均匀
B(omega) 是否非负或接近耗散
A(omega), B(omega), F_ex(omega) 是否有 NaN/Inf
高频截断是否平滑
低频端是否缺失
K_r(t) 是否快速衰减
K_r(t) 是否存在长期不衰减振荡
```

当前 adapter 中的水动力外推和 radiation kernel 诊断用于提高 Cummins 时域响应稳定性，但外推是 opt-in 行为，不会覆盖原始频域数据。

### 1.5 规则波和波谱激励

规则波激励：

```text
F_exc(t) = Re{ F_ex(omega) * a * exp(-i omega t + i phi) }
```

波谱激励：

```text
eta(t) = sum_j a_j cos(omega_j t + phi_j)
a_j = sqrt(2 S_eta(omega_j) Delta_omega_j)
F_exc(t) = sum_j Re{ F_ex(omega_j) * a_j * exp(-i omega_j t + i phi_j) }
```

当前支持：

```text
JONSWAP
Pierson-Moskowitz
random phase controlled by spectrum_seed
```

常用验证海况：

```text
Hs = 1.0 m
gamma = 3.3
omega_peak = 0.4157 rad/s 或其他 target omega
```

### 1.6 Direct Cummins 与 ERA 状态空间

#### Direct Cummins

Direct Cummins 使用离散卷积直接计算辐射记忆力：

```text
F_rad_memory[n] = sum_k K_r[k] qd[n-k] dt
```

优点：

```text
物理含义最清楚；
适合作为基准解；
适合检查 radiation kernel 和频域 RMS 闭合。
```

缺点：

```text
长时间计算需要反复计算历史卷积；
当 DOF 数和时间步数增加时成本较高。
```

#### ERA State-Space

状态空间方法把辐射记忆核拟合为离散状态空间模型：

```text
x_rad[n+1] = A_d x_rad[n] + B_d qd[n]
F_rad_memory[n+1] = C_d x_rad[n+1]
```

并保留 zero-lag kernel 以匹配直接卷积：

```text
F_rad_total = F_rad_memory + D_0 qd
```

当前推荐基准：

```text
state_order = 240
era_block_rows = 55
era_block_cols = 55
radiation_passivity_correction = clip_negative_eigenvalues
```

状态空间模型可以保存和复用：

```text
save_discrete_state_space_radiation_model(...)
load_discrete_state_space_radiation_model(...)
```

复用时必须保证：

```text
time_step 一致；
reduced DOF 数一致；
hydrodynamic dataset 一致；
master node / structural reduction 设置一致。
```

### 1.7 时间积分器

当前平台支持两个固定步长积分器：

```text
integrator = "newmark"
integrator = "rk4"
```

#### Newmark

默认生产积分器为 Newmark 平均加速度法：

```text
beta = 0.25
gamma = 0.5
```

特点：

```text
隐式格式；
对当前 reduced flexible hydroelastic system 更稳健；
推荐用于常规算例、长时间记录和多海况批量计算。
```

#### RK4

RK4 是显式四阶 Runge-Kutta，用作数值交叉验证：

```text
qdot = v
vdot = M_eff^{-1} [F(t) - F_memory(t) - C_eff v - K_eff q]
```

当前实现中：

```text
1. RK4 与 Newmark 使用同一套 reduced matrices；
2. direct Cummins 路径中，history force 仍按已知速度历史计算；
3. RK4 子步内对外力和 memory force 做线性插值；
4. state-space 路径中，ERA radiation state 在网格点更新，机械方程在步内使用 RK4 推进；
5. 求解结束后同样只在需要输出时还原 global response。
```

重要稳定性结论：

```text
Newmark 是当前默认和推荐生产格式；
RK4 是显式方法，对当前柔性降维系统的高频模态更敏感；
粗步长 RK4 可能溢出或 NaN；
RK4 使用前应先做短时间稳定性验证。
```

已验证的稳定 RK4 设置：

```text
steps_per_cycle = 400
memory_cycles = 2
state_order = 240
mooring horizontal stiffness = 1e7 N/m per corner/DOF
```

### 1.8 系泊接口

当前系泊不是最终物理系泊模块，只是线性化 stiffness provider 接口：

```python
MooringLinearization(
    reduced_stiffness=K_moor_reduced,
    metadata={...},
)
```

或者：

```python
provider(case, structural) -> MooringLinearization | ndarray | None
```

adapter 只接收 reduced stiffness，不要求 RODM 主程序了解系泊模型。

当前四角弹簧只是验证示例：

```text
corner nodes = 1, 61, 733, 793
k_surge = k_sway = 1e7 N/m
k_heave = 0
```

未来正式系泊模块只要输出 reduced stiffness 或可被投影为 reduced stiffness 的线性化结果，即可接入时域平台。

## 2. 代码实现步骤及逻辑

### 2.1 目录结构

核心时域代码：

```text
src/offshore_energy_sim/time_domain/
src/offshore_energy_sim/time_domain_adapter/
```

用户入口和验证脚本：

```text
scripts/run_wecsim_like_time_domain_platform.py
scripts/validate_wecsim_like_multi_sea_state.py
scripts/validate_time_integrator_comparison.py
scripts/validate_hydrodynamic_extrapolation.py
```

文档和结果：

```text
docs/time_domain/
results/time_domain/
```

### 2.2 主要模块职责

`src/offshore_energy_sim/time_domain/solver.py`

```text
solve_linear_time_domain(...)
solve_linear_time_domain_rk4(...)
solve_rodm_time_domain_case(...)
direct_convolution_memory_force(...)
```

职责：

```text
1. 线性 reduced system 时间积分；
2. Newmark 和 RK4 积分器选择；
3. direct Cummins 历史卷积；
4. 输出 reduced displacement/velocity/acceleration/memory force。
```

`src/offshore_energy_sim/time_domain/rodm_hydrodynamics.py`

```text
prepare_rodm_time_domain_hydrodynamic_terms(...)
```

职责：

```text
1. 从 RODM 频域水动力数据生成时域所需水动力项；
2. 计算 A_inf, residual terms, radiation_irf；
3. 生成规则波或波谱激励力时间序列。
```

`src/offshore_energy_sim/time_domain/excitation.py`

职责：

```text
1. JONSWAP/PM 波谱；
2. 随机相位；
3. 波面时间序列；
4. 多频波浪激励力合成。
```

`src/offshore_energy_sim/time_domain_adapter/wecsim_like_solver.py`

```text
solve_rodm_wecsim_like_time_domain(...)
WecSimLikeRadiationConfig(...)
WecSimLikeTimeDomainResult(...)
MooringLinearization(...)
```

职责：

```text
1. 外接 WEC-Sim-like 平台主入口；
2. 接收 RODM case、time-domain config、radiation config；
3. 解析 direct_convolution 或 state_space；
4. 接入 optional mooring provider；
5. 调用 reduced-space solver；
6. 最后还原 global response；
7. 保证 adapter 不反向污染 RODM 频域核心。
```

`src/offshore_energy_sim/time_domain_adapter/state_space_radiation.py`

职责：

```text
1. ERA 状态空间拟合；
2. passivity/spectral-radius 处理；
3. kernel 重构误差评估；
4. state-space model 保存和读取。
```

`src/offshore_energy_sim/time_domain_adapter/state_space_solver.py`

职责：

```text
1. 使用 ERA radiation state 求解 reduced linear system；
2. 支持 Newmark；
3. 支持 RK4；
4. 输出与 direct Cummins 一致的数据结构。
```

`src/offshore_energy_sim/time_domain_adapter/mooring.py`

职责：

```text
1. 提供线性化 reduced stiffness 接口；
2. 提供四角弹簧验证示例；
3. 不代表最终系泊模块。
```

`src/offshore_energy_sim/time_domain_adapter/hydrodynamic_extrapolation.py`

职责：

```text
1. 低频/高频水动力外推；
2. 保护原始频率范围内数据不变；
3. 为更稳定 radiation kernel 提供扩展频率数据。
```

### 2.3 主求解流程

平台主入口：

```python
solve_rodm_wecsim_like_time_domain(
    case,
    config,
    radiation=WecSimLikeRadiationConfig(...),
    mooring_provider=provider,
)
```

实际流程：

```text
1. 读取 RODM frequency case；
2. 打开 hydrodynamic dataset；
3. 执行或读取结构降维结果；
4. 在 reduced/master DOF 空间生成 M, K, A_inf, A_res, C_res；
5. 根据 B(omega) 生成 radiation_irf；
6. 根据规则波或波谱生成 F_exc(t)；
7. 可选接入 K_moor_reduced；
8. 选择 direct_convolution 或 state_space；
9. 选择 Newmark 或 RK4；
10. 在 reduced/master DOF 上进行时间推进；
11. 求解结束后还原 global retained DOF 响应；
12. 输出时间序列、RMS、图和指标文件。
```

伪代码：

```python
hydro = prepare_rodm_time_domain_hydrodynamic_terms(case, dataset, config)

M_eff = M_reduced + hydro.added_mass_infinite + hydro.residual_added_mass
C_eff = hydro.residual_radiation_damping
K_eff = K_struct_reduced + K_hs_reduced + K_moor_reduced
F_time = hydro.excitation_force

if radiation.model == "direct_convolution":
    reduced = solve_linear_time_domain(
        mass=M_eff,
        damping=C_eff,
        stiffness=K_eff,
        force=F_time,
        radiation_irf=hydro.radiation_irf,
        integrator=radiation.integrator,
    )
else:
    ss = fit_or_load_era_state_space_model(hydro.radiation_irf)
    reduced = solve_state_space_radiation_linear_system(
        system,
        integrator=radiation.integrator,
    )

global_displacement = reconstruct_global_response(T, reduced.displacement)
```

### 2.4 积分器选择

在 Python API 中：

```python
WecSimLikeRadiationConfig(
    model="direct_convolution",
    integrator="newmark",
)

WecSimLikeRadiationConfig(
    model="state_space",
    integrator="rk4",
)
```

在命令行中：

```powershell
--integrator newmark
--integrator rk4
```

推荐：

```text
常规计算、长时间记录、多海况：newmark
短时间交叉验证、积分器敏感性检查：rk4
```

### 2.5 输出对象

`WecSimLikeTimeDomainResult` 主要包含：

```text
time
master_displacement
master_velocity
master_acceleration
global_displacement
memory_force
wave_elevation
excitation_force
radiation_model
integrator
radiation_irf_time
radiation_irf
added_mass_infinite
residual_added_mass
residual_radiation_damping
state_space_model
mooring_reduced_stiffness
mooring_metadata
```

这意味着后续光伏发电、运动损失、长期能量评估模块可以直接使用：

```text
global_displacement(t)
centerline heave(t)
selected node motion(t)
wave_elevation(t)
```

## 3. 用户指南

### 3.1 环境准备

进入仓库：

```powershell
cd C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最近一次全量测试记录：

```text
83 passed
```

最近一次与 Newmark/RK4 相关的定向测试：

```text
36 passed
```

### 3.2 数据路径

默认本地标准化数据路径：

```text
data/external/DM-FEM2D/
```

主要水动力数据：

```text
data/external/DM-FEM2D/HydrodynamicData/Yoga/DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
```

主要结构矩阵：

```text
data/external/DM-FEM2D/StructureData/JobMesh5_5_MASS1.mtx
data/external/DM-FEM2D/StructureData/JobMesh5_5_STIF1.mtx
```

如果使用外部数据源，可指定：

```powershell
--data-root E:\phd\Code\DM-FEM2D
```

或设置环境变量：

```powershell
$env:RODM_DM_FEM_ROOT = "E:\phd\Code\DM-FEM2D"
```

### 3.3 运行单个 WEC-Sim-like 算例

推荐先运行 direct Cummins 和 state-space 双路径对比：

```powershell
.\.venv\Scripts\python.exe scripts\run_wecsim_like_time_domain_platform.py `
  --output-root results\time_domain\wecsim_like_platform_dm10 `
  --radiation-model both `
  --integrator newmark `
  --cycles 40 `
  --steps-per-cycle 50 `
  --memory-cycles 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000 `
  --save-state-space-model-path results\time_domain\wecsim_like_platform_dm10\state_space_radiation_era240.npz `
  --save-arrays
```

主要输出：

```text
wecsim_like_platform_metrics.json
wecsim_like_platform_summary.csv
report.md
arrays/
figures/
```

常看图：

```text
figures/direct_vs_state_centerline_heave_time.png
figures/direct_vs_state_centerline_heave_rms.png
figures/radiation_memory_force_norm.png
```

### 3.4 运行 Direct Cummins

```powershell
.\.venv\Scripts\python.exe scripts\run_wecsim_like_time_domain_platform.py `
  --output-root results\time_domain\direct_only_case `
  --radiation-model direct_convolution `
  --integrator newmark `
  --cycles 40 `
  --steps-per-cycle 50 `
  --memory-cycles 2 `
  --mooring-corner-horizontal-stiffness 10000000
```

适用于：

```text
1. 建立基准解；
2. 检查 Cummins memory kernel；
3. 做频域 RMS 和时域 RMS 闭合验证。
```

### 3.5 运行 State-Space

如果已经保存 ERA 模型：

```powershell
.\.venv\Scripts\python.exe scripts\run_wecsim_like_time_domain_platform.py `
  --output-root results\time_domain\state_space_only_case `
  --radiation-model state_space `
  --integrator newmark `
  --state-space-model-path results\time_domain\wecsim_like_platform_dm10\state_space_radiation_era240.npz `
  --cycles 120 `
  --steps-per-cycle 50 `
  --memory-cycles 2 `
  --mooring-corner-horizontal-stiffness 10000000
```

适用于：

```text
1. 长时间时程；
2. 多海况扫描；
3. 后续光伏发电和运动损失评估。
```

### 3.6 运行 RK4 交叉验证

RK4 不建议直接用于粗步长长时间生产计算。建议先做短时间、细步长验证：

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_integrator_comparison.py `
  --output-root results\time_domain\time_integrator_newmark_rk4_timeseries_spc400 `
  --cycles 6 `
  --ramp-cycles 1 `
  --steps-per-cycle 400 `
  --memory-cycles 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

输出：

```text
time_integrator_comparison_metrics.json
figures/newmark_vs_rk4_centerline_heave_rms.png
figures/direct_convolution_newmark_vs_rk4_heave_time.png
figures/state_space_newmark_vs_rk4_heave_time.png
figures/newmark_vs_rk4_error_norm_time.png
```

最新时间序列验证结果：

```text
case: JONSWAP, Hs = 1.0 m, omega_peak = 0.4157 rad/s
cycles = 6
steps_per_cycle = 400
memory_cycles = 2
state_order = 240
mooring horizontal stiffness = 1e7
```

Newmark 为 reference，RK4 为 candidate：

| Radiation model | Master displacement L2 | Memory force L2 | Global displacement L2 | Centerline heave L2 | Centerline heave RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct Cummins | 4.334e-4 | 1.853e-4 | 4.273e-4 | 7.371e-5 | 3.651e-5 |
| ERA state-space | 4.120e-4 | 2.252e-4 | 4.070e-4 | 8.857e-5 | 4.932e-5 |

时间序列图位置：

```text
results/time_domain/time_integrator_newmark_rk4_timeseries_spc400/figures/direct_convolution_newmark_vs_rk4_heave_time.png
results/time_domain/time_integrator_newmark_rk4_timeseries_spc400/figures/state_space_newmark_vs_rk4_heave_time.png
results/time_domain/time_integrator_newmark_rk4_timeseries_spc400/figures/newmark_vs_rk4_error_norm_time.png
```

解释：

```text
Newmark 与 RK4 的代表性中心线 heave 时间历程基本重合；
没有看到明显相位漂移；
误差主要随时间轻微累积，但仍维持在 1e-3 以下的相对范数；
说明 reduced-space RK4 实现可作为积分器交叉验证。
```

### 3.7 多海况验证

短记录多海况 state/direct 回归：

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py `
  --output-root results\time_domain\wecsim_like_multi_sea_state_validation `
  --hs-values 0.5,1.0 `
  --target-omega-values 0.35,0.4157,0.55 `
  --seeds 1 `
  --cycles 20 `
  --steps-per-cycle 40 `
  --memory-cycles 2 `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

长记录频域 RMS 闭合：

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py `
  --output-root results\time_domain\wecsim_like_long_direct_frequency_rms_closure `
  --hs-values 1.0 `
  --target-omega-values 0.4157,0.70 `
  --seeds 1,2 `
  --cycles 120 `
  --steps-per-cycle 40 `
  --memory-cycles 2 `
  --skip-long-run `
  --state-order 240 `
  --era-block-rows 55 `
  --era-block-cols 55 `
  --mooring-corner-horizontal-stiffness 10000000
```

已获得的长记录验证指标：

```text
cases = 4
cycles = 120
samples per case = 4801
max frequency/direct fitted RMS error = 7.923e-3
max frequency/state fitted RMS error = 1.072e-2
max state/direct centerline heave RMS error = 4.102e-3
```

这说明：

```text
Direct Cummins 与频域 RMS 可以在长记录下闭合到约 1%；
ERA-240 状态空间模型可以作为长时间计算的实用辐射模型；
短随机波记录不适合直接要求频域理论 RMS 完全闭合，应使用长记录或多 seed 平均。
```

### 3.8 关键参数建议

#### 时间步长

Newmark：

```text
--steps-per-cycle 40 to 60
```

RK4：

```text
--steps-per-cycle 400 起步做稳定性检查
```

#### 记忆时长

```text
--memory-cycles 2
```

当前 DM10 dense-88 数据下推荐保持 2。过短会截断 radiation memory，过长可能引入有限频带尾部振荡。

#### 卷积规则

```text
--radiation-convolution-rule trapezoidal
```

推荐使用 `trapezoidal`。

#### residual 修正

```text
--radiation-residual-model selected_frequency
```

用于让有限频带 Cummins 时域模型在目标频率处更接近原频域水动力。

#### ERA 参数

```text
--state-order 240
--era-block-rows 55
--era-block-cols 55
```

当前 DM10 基准推荐该组合。

#### 系泊参数

```text
--mooring-corner-horizontal-stiffness 10000000
```

这只是线性 stationkeeping 验证项，不代表最终系泊模块。

### 3.9 如何判断结果可信

优先检查：

```text
state_vs_direct_centerline_heave_rms_relative_error
state_vs_direct_memory_force_l2_relative_error
frequency_vs_direct_fit_rms_error
frequency_vs_state_fit_rms_error
wave_elevation_reconstructed_hs
state_space spectral_radius
Newmark/RK4 centerline heave time-history overlap
```

建议阈值：

```text
短记录 state/direct heave RMS error < 1%
长记录 frequency/direct fitted RMS error < 1% to 2%
长记录 frequency/state fitted RMS error < 1% to 2%
Newmark/RK4 heave RMS error 在细步长下约 1e-4 或更小
reconstructed Hs 接近 target Hs
state-space spectral radius <= 1
```

如果频域 RMS 和时域 RMS 差异较大，优先排查：

```text
1. 记录是否太短；
2. ramp 后有效样本是否不足；
3. 随机 seed 是否导致 reconstructed Hs 偏离；
4. 波谱频率点是否足够；
5. 是否错误地用短记录要求理论 RMS 闭合；
6. radiation kernel 是否存在长时间不衰减振荡；
7. RK4 是否使用了过粗步长。
```

### 3.10 常见问题

#### Direct Cummins 报错：requires at least two frequencies

说明当前 hydrodynamic 文件只有单频数据，不能生成 radiation kernel。请使用多频 dense 数据：

```text
DM10_direction0_cummins_spectrum_dense_88_mesh2.nc
```

#### State-space 模型 time step 不匹配

ERA 模型保存时包含 `time_step`。如果修改了 `steps-per-cycle`、目标频率、结构降维或水动力数据，需要重新拟合状态空间模型。

#### RK4 结果 NaN 或发散

RK4 是显式方法，粗步长下可能不稳定。处理方式：

```text
1. 增大 steps_per_cycle，例如 400；
2. 缩短 cycles 做稳定性验证；
3. 对生产计算使用 Newmark；
4. 检查 stiffness 和 high-frequency flexible modes。
```

#### Master displacement error 大于 heave RMS error

master norm 可能包含 surge/sway/rotation 等分量，受低频漂移和 stationkeeping 设置影响。大型 OFPV 的水弹性验证中，中心线 heave RMS、代表点 heave 时间序列和运动谱通常是更直接的观测量。

#### 长时间运行太慢

建议：

```text
1. 用 Direct Cummins 做少量基准；
2. 保存 ERA state-space 模型；
3. 后续长时间和多海况扫描使用 state_space；
4. 只在需要时保存 arrays。
```

## 4. 最新完成度

### 已完成

```text
1. 外接 WEC-Sim-like 时域平台；
2. Cummins 直接卷积；
3. ERA 状态空间辐射近似；
4. 规则波和波谱输入；
5. 多频激励力合成；
6. reduced/master DOF 时间推进；
7. global retained DOF 后处理还原；
8. Newmark 默认积分器；
9. RK4 显式四阶积分器；
10. Newmark/RK4 时间序列对比图；
11. 简单线性 reduced mooring provider 接口；
12. ERA 模型保存和复用；
13. 多海况 state/direct 验证；
14. 长记录 frequency/time RMS 闭合验证；
15. hydrodynamic extrapolation 和 radiation kernel 诊断。
```

### 保持的架构边界

```text
RODM frequency-domain core modified = false
RODM 主程序不依赖 time_domain_adapter
time_domain_adapter 只读取 RODM 频域结果
系泊仍通过外部 provider 接入
Newmark/RK4 都在 reduced/master DOF 上推进
global response 只在输出阶段还原
```

### 尚未完成但已预留接口

```text
1. 正式非线性系泊模块；
2. PTO/control；
3. 非线性水动力；
4. 二阶漂移力；
5. 更大规模海况数据库；
6. 与外部 WEC-Sim 刚体 benchmark 的逐项对照。
```

## 5. 快速命令索引

全量测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

单算例 direct/state 对比：

```powershell
.\.venv\Scripts\python.exe scripts\run_wecsim_like_time_domain_platform.py --radiation-model both --cycles 40 --steps-per-cycle 50 --state-order 240 --mooring-corner-horizontal-stiffness 10000000
```

RK4/Newmark 时间序列对比：

```powershell
.\.venv\Scripts\python.exe scripts\validate_time_integrator_comparison.py --cycles 6 --ramp-cycles 1 --steps-per-cycle 400 --state-order 240 --mooring-corner-horizontal-stiffness 10000000
```

多海况短记录验证：

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py --hs-values 1.0 --target-omega-values 0.30,0.35,0.4157,0.55,0.70 --seeds 1,2 --cycles 20 --steps-per-cycle 40 --state-order 240 --mooring-corner-horizontal-stiffness 10000000
```

长记录频域 RMS 闭合：

```powershell
.\.venv\Scripts\python.exe scripts\validate_wecsim_like_multi_sea_state.py --hs-values 1.0 --target-omega-values 0.4157,0.70 --seeds 1,2 --cycles 120 --steps-per-cycle 40 --skip-long-run --state-order 240 --mooring-corner-horizontal-stiffness 10000000
```

## 6. 结论

当前时域平台已经形成了清晰的 WEC-Sim-like 外接时域框架：

```text
RODM 频域模型负责水弹性主求解；
time_domain_adapter 负责 Cummins/状态空间时域适配；
Newmark 负责稳健生产计算；
RK4 负责细步长交叉验证；
所有推进先在 reduced/master DOF 完成；
global response 只在输出和后处理时还原。
```

从现有验证看，Direct Cummins、ERA state-space、频域 RMS、时域 RMS、Newmark 和 RK4 之间已经建立了基本闭合关系。后续最适合继续推进的是：

```text
1. 固化 2 到 4 个长记录 benchmark；
2. 用 state-space 进行更大范围 Hs/Tp/seed 扫描；
3. 接入正式系泊模块的 provider；
4. 将 global motion time series 传递给光伏发电和运动损失模型。
```
