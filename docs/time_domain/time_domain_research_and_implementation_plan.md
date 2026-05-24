# 时域水弹性模拟调研与实现计划

日期：2026-05-21

## 1. 结论摘要

当前代码库已经具备进入时域模拟的关键前置条件：结构矩阵读取、5DOF 保留约定、SEREP/Guyan 缩减、Capytaine 频域水动力数据读取、规则波/随机波谱、风谱、连接件力恢复和标准化验证脚本。但当前主线仍是频域方法，还没有真正的时域求解器、辐射记忆函数、状态空间辐射模型或时间积分工作流。

推荐下一步不要直接写“大而全”的非线性时域平台，而是先实现一个线性、缩减坐标下的 Cummins 方程求解器：

```text
(M_s + A_inf) q_ddot(t)
+ C_ext q_dot(t)
+ K_total q(t)
+ integral_0^t K_rad(t - tau) q_dot(tau) d tau
= F_exc(t) + F_wind(t) + F_other(t)
```

其中 `q` 是当前 RODM 主自由度坐标，维度通常为 `hydrodynamic_nodes * retained_dofs_per_node`，例如 10 个水动力节点 x 5DOF = 50DOF。结构侧继续复用当前 `prepare_structural_reduction`，水动力侧从 Capytaine 输出的 `added_mass(omega)`、`radiation_damping(omega)`、`diffraction_force`、`Froude_Krylov_force` 生成时域所需的 `A_inf`、辐射记忆核 `K_rad(t)` 和激励力时间序列。

建议分三步实施：

1. 先做无辐射记忆或常系数近似的线性时域稳态验证，保证时间积分、相位约定和频域结果一致。
2. 再加入 IRF 直接卷积，建立 Cummins 方程的正确性验证。
3. 最后实现 WEC-Sim 类似的状态空间辐射模型，用于生产级 50DOF/100DOF 长时间模拟。

## 2. 本地代码基础

### 2.1 已经标准化的可复用模块

当前频域平台的主线已经很适合时域扩展：

- `src/offshore_energy_sim/core/cases.py`：已有 `RodmFrequencyCase`，可仿照建立 `RodmTimeDomainCase`。
- `src/offshore_energy_sim/structure/rodm_reduction.py`：读取结构质量/刚度矩阵，删除第 6 自由度，生成缩减结构质量/刚度和变换矩阵。时域仍应复用。
- `src/offshore_energy_sim/hydrodynamics/frequency.py`：已实现单频水动力矩阵和波浪力整理，可扩展为多频/时域预处理。
- `src/offshore_energy_sim/solver/frequency_domain.py`：频域解可作为时域稳态结果的验证基准。
- `src/offshore_energy_sim/environment/spectra.py`：已有 JONSWAP 波谱、API 风谱、谱到幅值转换。
- `src/offshore_energy_sim/response/spectral.py`：已有响应谱/RMS 工具，可用于随机波时域结果对照。
- `src/offshore_energy_sim/hydrodynamics/capytaine_array.py`：已有 Capytaine 阵列水动力生成器和 UI，可用于生成更密集频率网格。
- `src/offshore_energy_sim/strength/connector_recovery.py`：已有频域连接件相对位移和力恢复概念，时域中可扩展为 `F_c(t)=K delta(t)+C delta_dot(t)`。

### 2.2 过去研究中的有用线索

本地历史文件中没有发现完整的时域 Cummins 求解器，但有几类重要前置工作：

- `RODM_uneven_wave.ipynb`：做过不均匀/组合水动力数据集和多频率频域响应研究。它仍是频域逐频求解，但说明已有“多频数据集 + 响应谱”思路。
- `RODM_WindStudy.ipynb`、`RODM_WindStudy_vv3_improve.ipynb`、`RODM_Wind_main.py`：做过风浪频域耦合。核心逻辑是 JONSWAP/API 谱得到幅值，再对每个频率求解响应，最后计算 RMS 或绘图。
- `DM_Windload.py`：已有风速剖面、API 风谱、风力幅值、风阻尼、模块集总风载、沿风向相位延迟。这个相位延迟思想可迁移到时域风荷载生成。
- `RODM_force_1.ipynb` 和 `src/offshore_energy_sim/strength`：已有从响应恢复模块内力/界面力的基础。时域中可按每个时间步恢复连接件力、弯矩包络和疲劳前处理量。

这些研究说明：本库目前更接近“频域谱分析 + 统计响应”，下一步时域化的关键不是重写结构/水动力读取，而是补齐“频域水动力到时域记忆核”和“时间积分器”。

## 3. 外部方法调研

### 3.1 Capytaine

Capytaine 的定位是频域 BEM 求解器，主要输出附加质量、辐射阻尼、静水恢复力、Froude-Krylov 力和绕射力等频域水动力系数。当前本库也正是通过 Capytaine/xarray NetCDF 数据驱动 RODM 频域求解。

调研结论：

- Capytaine 适合作为时域模型的频域水动力数据来源。
- Capytaine 本身不等同于完整时域运动求解器；它不替代 Cummins 方程积分器。
- 下一步应让 Capytaine 输出覆盖足够宽的频率范围，并尽量包含高频/无限频率附加质量所需信息。
- 当前 `run_hydrodynamics_ui.py` 已能生成多频率 NetCDF，是时域前处理的重要入口。

需要重点确认的 Capytaine 约定：

- `added_mass(omega)` 与 `radiation_damping(omega)` 的单位和 DOF 标签顺序。
- `omega` 网格是否足够覆盖 IRF 变换需要的低频和高频段。
- 是否可以直接计算或近似 `A_inf`。若不能可靠得到，应先用高频 `added_mass` 或 IRF 关系估计，并做验证。
- `Froude_Krylov_force + diffraction_force` 的复相位约定，需要通过规则波时域稳态对照频域解确认。

### 3.2 WEC-Sim

WEC-Sim 是成熟的时域波能装置仿真框架。它的核心思想非常适合借鉴，但不能直接照搬到本库：

- WEC-Sim 的时域方程基于 Cummins 方程。
- 辐射力通常拆为无限频率附加质量项和辐射记忆卷积项。
- BEMIO 预处理会根据频域辐射阻尼生成 IRF，并可进一步拟合状态空间模型。
- 生产仿真中常使用状态空间辐射模型替代直接卷积，以提升长时间仿真性能。
- WEC-Sim 的力学框架面向刚体/多体 WEC，当前本库是结构 FEM 缩减后的柔性水弹性 RODM；因此应借鉴水动力时域化方法，而不是照搬 Simulink 架构。

对本库最有价值的 WEC-Sim 思路：

1. 建立独立的 BEM 数据预处理层：频域系数 -> `A_inf`、`K_rad(t)`、激励力插值。
2. 支持两种辐射模型：直接卷积和状态空间。
3. 以规则波稳态频域解作为时域验证基准。
4. 保留不同选项：常系数近似、IRF 卷积、状态空间近似，便于逐步验证。

## 4. 推荐数学模型

### 4.1 缩减坐标时域方程

当前频域方程是：

```text
(-omega^2 M_eff - i omega C_eff + K_eff) X = F
```

时域中推荐在同一组主自由度坐标 `q(t)` 下求解：

```text
(M_R + A_inf) q_ddot
+ C_user q_dot
+ (K_R + K_hydrostatic + K_connector + K_mooring) q
+ F_rad_memory
= F_exc + F_wind + F_external
```

辐射记忆力：

```text
F_rad_memory(t) = integral_0^t K_rad(t - tau) q_dot(tau) d tau
```

其中：

- `M_R`、`K_R` 来自结构缩减。
- `A_inf` 是无限频率附加质量，和结构质量合并到惯性项。
- `K_hydrostatic` 来自 Capytaine 静水恢复力。
- `C_user` 可先为空或放结构阻尼/风阻尼/经验阻尼。
- `F_exc` 来自 Froude-Krylov + diffraction 的时域合成。
- `F_wind` 来自已有 API 风谱和空间相位模型扩展。

### 4.2 由频域阻尼生成辐射记忆核

若使用实际单位的辐射阻尼 `B(omega)`，推荐采用：

```text
K_rad(t) = 2 / pi * integral_0^infty B(omega) cos(omega t) d omega
```

数值实现：

```text
K_rad[t_index, :, :] =
    2/pi * trapz(B[omega_index, :, :] * cos(omega * t), omega)
```

注意：WEC-Sim/BEMIO 内部可能使用归一化阻尼，所以源码中会出现额外的 `omega` 或 `rho` 因子。本库应以 Capytaine 输出的实际矩阵单位为准，不能机械套用 WEC-Sim 归一化公式。

### 4.3 无限频率附加质量

优先级：

1. 若 Capytaine/数据集可直接给出或计算 `A_inf`，优先使用直接结果。
2. 否则使用高频端 `A(omega_max)` 作为初始近似。
3. 再进一步用 IRF 关系校正：

```text
A_inf ~= A(omega) + 1/omega * integral_0^infty K_rad(t) sin(omega t) d t
```

实际实现中应记录 `A_inf_method`，例如 `direct`, `high_frequency_limit`, `irf_consistency_fit`。

### 4.4 激励力时间序列

规则波：

```text
F_exc(t) = Re{ F_exc_hat(omega, theta) * a * exp(i omega t) }
```

随机波：

```text
F_exc(t) = Re{ sum_j F_exc_hat(omega_j, theta)
                   * sqrt(2 S_eta(omega_j) Delta_omega)
                   * exp(i (omega_j t + phi_j)) }
```

其中 `phi_j` 是随机相位。为了验证和复现，必须支持随机种子。初期先做单方向长峰不规则波；多方向波谱可作为后续扩展。

## 5. 推荐代码结构

建议新增以下源码包，但第一步先写文档和测试设计：

```text
src/offshore_energy_sim/time_domain/
├── __init__.py
├── cases.py
├── hydrodynamic_memory.py
├── excitation.py
├── integrators.py
├── radiation.py
├── solver.py
└── postprocess.py
```

职责建议：

- `cases.py`：定义 `RodmTimeDomainCase`，包含时间步长、总时长、辐射模型、波况、阻尼、输出控制。
- `hydrodynamic_memory.py`：从 Capytaine 多频数据生成 `A_inf`、`K_rad(t)`、IRF 诊断图和频域一致性误差。
- `excitation.py`：生成规则波/随机波的波面和激励力时间序列。
- `radiation.py`：定义辐射模型接口，包括 `NoMemoryRadiation`、`DirectConvolutionRadiation`、`StateSpaceRadiation`。
- `integrators.py`：固定步长 Newmark/RK4/半隐式积分器。初期建议固定步长，便于卷积和验证。
- `solver.py`：组织结构缩减、水动力时域化、载荷生成和时间积分。
- `postprocess.py`：时间序列转稳态幅值、RMS、PSD、连接件力包络、动画/图件输出。

也可把代码放入现有 `solver/`、`hydrodynamics/`、`environment/`，但独立 `time_domain/` 更利于隔离风险。

## 6. 实现阶段计划

### 阶段 0：补充数据和约定检查

目标：确认时域所需数据是否完整。

任务：

- 用现有 UI 或脚本生成一个宽频率网格 NetCDF，例如 `omega=0.05..3.0 rad/s`，至少 100-200 个点。
- 检查 `added_mass`、`radiation_damping`、`hydrostatic_stiffness`、`Froude_Krylov_force`、`diffraction_force` 的维度和 DOF 标签。
- 明确 `reverse_hydrodynamic_node_order`、水动力节点顺序、结构主节点顺序在多频数据中的一致性。
- 写 `docs/time_domain/equation_conventions.md`，冻结相位、单位和 DOF 顺序。

验收：

- 能从同一份多频 NetCDF 复现当前一个频率点的频域结果。
- 规则波单频时域输入的相位约定与频域响应一致。

### 阶段 1：无记忆线性时域原型

目标：先让时域积分框架跑通，不引入 IRF 复杂性。

方程：

```text
(M_R + A(omega_ref)) q_ddot + B(omega_ref) q_dot + K q = F_exc(t)
```

任务：

- 建立 `RodmTimeDomainCase` 和 `solve_rodm_time_domain_case` 原型。
- 使用规则波激励 `F_exc(t)`。
- 使用 `solve_ivp` 或固定步长 RK4。
- 输出 `q(t)`、`q_dot(t)`、`q_ddot(t)` 和重构后的 `global_response(t)`。

验收：

- 对单频规则波，去掉初始瞬态后，用正弦拟合得到的稳态幅值应接近当前频域 `solve_rodm_frequency_case`。
- 先允许 1-5% 误差，稳定后收紧。

### 阶段 2：IRF 直接卷积 Cummins 方程

目标：加入真实辐射记忆。

任务：

- 从 `B(omega)` 生成 `K_rad(t)`。
- 实现固定步长历史卷积：

```text
F_memory[n] ~= sum_m K_rad[m] @ q_dot[n-m] * dt
```

- 添加记忆截断时间 `memory_time_s` 和窗口函数选项。
- 输出 IRF 图、截断误差、由 IRF 回算频域阻尼的误差。

验收：

- 1DOF 人工数据可通过解析/数值反演验证。
- 规则波稳态响应与频域解接近。
- 随机波 RMS 与频域谱积分结果趋势一致。

### 阶段 3：状态空间辐射模型

目标：解决直接卷积在 50DOF/100DOF 长时间仿真中的性能问题。

推荐形式：

```text
x_rad_dot = A_rad x_rad + B_rad q_dot
F_memory = C_rad x_rad + D_rad q_dot
```

任务：

- 先做逐 DOF-pair 的低阶状态空间拟合，或对 IRF 矩阵做 SVD 降维后拟合。
- 提供拟合阶数、误差、稳定性检查。
- 允许和直接卷积结果对照。

验收：

- 状态空间输出的 `F_memory` 与直接卷积足够接近。
- 长时间随机波模拟速度明显优于直接卷积。

### 阶段 4：风浪耦合与连接件时域后处理

目标：把过去风浪频域研究迁移到时域。

任务：

- 将 `DM_Windload.py` 中的 API 风谱、空间相位延迟、集总模块风载迁移为标准 `loads/time_domain_wind.py`。
- 支持 `F_wind(t)` 和线性化风阻尼。
- 扩展连接件力恢复：`F_connector(t)=K delta(t)+C delta_dot(t)`。
- 输出连接件剪力、弯矩、释放转角力矩的时域峰值、RMS 和疲劳预处理量。

验收：

- 风浪联合时域结果与现有频域风浪 RMS 结果量级一致。
- 连接件包络能和当前频域 `connector_recovery.py` 在单频稳态下对齐。

## 7. 风险和注意事项

高风险点：

- 相位约定：频域到时域的 `exp(i omega t)` 或 `exp(-i omega t)` 必须用规则波稳态对照验证。
- `A_inf`：不可靠的无限频率附加质量会直接影响加速度和固有频率。
- IRF 截断：记忆时间太短会改变阻尼，太长会性能很差。
- 频率网格：IRF 需要比当前单频/少频验证更宽更密的 `omega`。
- DOF 顺序：必须继承当前 one-based Abaqus 节点、5DOF 保留、水动力节点反序等约定。
- 柔性结构和刚体 WEC 的差异：WEC-Sim 方法可借鉴，但本库的坐标是 FEM 缩减坐标，不是简单 6DOF 刚体坐标。

数值结果预期：

- 新增文档不改变任何数值结果。
- 阶段 1 的常系数时域模型是近似模型，数值结果会有意区别于完整频域辐射模型。
- 阶段 2/3 目标是让规则波稳态与频域解一致，随机波统计量与频域谱积分一致。

## 8. 建议优先落地的最小可行版本

MVP 建议只覆盖：

- 300 m x 60 m 连续体参考算例；
- 10 个水动力主节点，5DOF，共 50DOF；
- 规则波单频；
- 常系数 `A(omega_ref)`、`B(omega_ref)`；
- 固定步长积分；
- 和频域稳态响应对比。

这个 MVP 不解决辐射记忆，但能最快验证：

- 数据接口；
- 时间积分；
- 激励相位；
- 响应重构；
- 后处理输出。

MVP 通过后，再加入 IRF 和状态空间。这样风险最低，也最符合当前仓库“先验证、再扩展”的数值安全原则。

## 9. 参考资料

- Capytaine 官方文档与代码库：`https://capytaine.org/`，`https://github.com/capytaine/capytaine`
- WEC-Sim 官方理论文档：`https://wec-sim.github.io/WEC-Sim/main/theory/theory.html`
- WEC-Sim GitHub 代码库：`https://github.com/WEC-Sim/WEC-Sim`
- WEC-Sim BEMIO：`https://wec-sim.github.io/bemio/`
- 本地相关文件：`RODM_uneven_wave.ipynb`、`RODM_WindStudy.ipynb`、`RODM_Wind_main.py`、`DM_Windload.py`、`src/offshore_energy_sim/environment/spectra.py`、`src/offshore_energy_sim/hydrodynamics/capytaine_array.py`、`src/offshore_energy_sim/solver/frequency_domain.py`
