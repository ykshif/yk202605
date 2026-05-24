# Mooring 板块 TODO List

日期：2026-05-24

目标：把当前 WEC-Sim Mooring Matrix 风格的基础线性系泊框架，推进成可配置、可验证、可接入主频域/时域流程，并为后续查表系泊、MoorDyn 或非线性系泊线动力学保留清晰接口。

## 1. 第一阶段：线性系泊核心闭环

- [x] 新建正式系泊包 `src/offshore_energy_sim/mooring/`。
- [x] 实现 `LinearMooringMatrix`，支持 `K_moor / C_moor / F0`。
- [x] 实现 WEC-Sim 风格线性力公式：

```text
F_moor = F0 - K_moor q - C_moor qdot
```

- [x] 实现 `NodalMooringAttachment`，把系泊矩阵绑定到 one-based 结构节点。
- [x] 实现 nodal attachment 到 natural retained global DOF 的装配。
- [x] 实现 natural retained global DOF 到 SEREP/Guyan reduced DOF 的投影。
- [x] 适配 WEC-Sim-like 时域求解器，使其接收 reduced `K/C/F0`。
- [x] 保持旧四角弹簧 adapter 接口可用。
- [x] 添加基础单元测试。
- [x] 编写系泊总文档 `docs/mooring_framework_cn.md`。

## 2. 第二阶段：配置驱动

- [x] 在 `configs/templates/rodm_frequency_case.yaml` 中加入 `mooring` 配置模板。
- [x] 添加 `configs/templates/mooring_linear_demo.yaml` 作为独立 demo YAML。
- [x] 支持 `mooring.enabled` 开关。
- [x] 支持 `mooring.model: linear_matrix`。
- [x] 支持单个或多个 `attachments`。
- [x] 支持 `node_one_based` 绑定。
- [x] 支持 `stiffness.diagonal` 输入。
- [x] 支持 `stiffness.matrix` 完整 6x6 输入。
- [x] 支持 `damping.diagonal` 输入。
- [x] 支持 `damping.matrix` 完整 6x6 输入。
- [x] 支持 `pretension` 1x6 输入。
- [x] 支持默认 retained full DOF 自动推断：

```python
retained = all_full_dofs - removed_full_dofs_zero_based
```

- [x] 新增 `src/offshore_energy_sim/mooring/config.py`。
- [x] 实现 `build_mooring_attachments_from_config(config)`。
- [x] 实现 `build_mooring_provider_from_config(config)`。
- [x] 实现 `build_reduced_mooring_terms_from_config(config, case, structural)`。
- [x] 为错误配置提供清晰报错，例如矩阵维度、节点越界、非数值输入。
- [x] 添加配置驱动测试 `tests/test_mooring_config.py`。

## 3. 第三阶段：接入主运行入口

- [x] 在 `scripts/run_rodm_case_from_config.py` 中读取 `mooring` 配置。
- [x] 对 `--domain time` 接入 mooring provider。
- [x] 对 WEC-Sim-like 平台入口接入正式 `offshore_energy_sim.mooring` provider。
- [x] 保留旧命令行四角弹簧参数，但在总文档中标记为简化 adapter example。
- [x] 输出 metrics 时记录系泊摘要：
  - enabled
  - model
  - attachment count
  - node ids
  - stiffness norm / trace
  - damping norm / trace
  - pretension norm
- [x] 保存 reduced `K_moor`、`C_moor`、`F0_moor` 到结果目录，便于复查。
- [x] 确认默认 `mooring.enabled: false` 时不创建 provider，现有路径不叠加系泊项。

## 4. 第四阶段：频域接入策略

- [ ] 明确频域中系泊的默认策略：
  - `K_moor` 可进入频域动态刚度。
  - `C_moor` 可进入频域阻尼项。
  - `F0` 默认不进入 RAO 谐波右端项。
- [ ] 在文档中说明 `F0` 与静平衡/均值漂移的关系。
- [ ] 为 `solve_rodm_frequency_case` 设计可选 mooring 参数，但默认不改变现有路径。
- [ ] 实现频域 reduced `K/C` 叠加。
- [ ] 添加频域线性系泊测试：
  - 无系泊结果不变。
  - 纯 `K_moor` 会改变固有恢复刚度。
  - 纯 `C_moor` 会改变频域阻尼响应。
- [ ] 给配置增加：

```yaml
mooring:
  apply_to_frequency_domain: false
  include_pretension_in_frequency_rhs: false
```

## 5. 第五阶段：时域验证

- [x] 构造 1DOF 解析 benchmark：
  - 质量
  - 结构刚度
  - mooring stiffness
  - mooring damping
  - harmonic forcing
- [x] 验证时域稳态幅值与频域解析解一致。
- [x] 构造 2DOF 带耦合 `K_moor/C_moor` 的 benchmark。
- [x] 验证 `F0` 会产生正确的静态偏置：

```text
q_static = K_eff^{-1} F0
```

- [x] 生成基础验证脚本 `scripts/validate_mooring_linear_framework.py`。
- [x] 生成基础验证报告 `docs/mooring_linear_framework_validation_2026_05_24.md`。
- [x] 输出基础验证 metrics `results/mooring/linear_framework_validation/metrics.json`。
- [x] 输出基础验证图片 `results/mooring/linear_framework_validation/figures/`：
  - `linear_mooring_error_summary.png`
  - `sdof_time_frequency_comparison.png`
  - `sdof_complex_amplitude_comparison.png`
  - `force_formula_comparison.png`
  - `nodal_assembly_error.png`
  - `reduced_projection_comparison.png`
  - `config_provider_comparison.png`
  - `coupled_2dof_complex_amplitude_comparison.png`
- [ ] 对 300 m RODM case 做无系泊/四角系泊/正式线性系泊三组对比。
- [ ] 比较 direct convolution 与 state-space radiation 在有系泊下的响应误差。
- [ ] 记录 centerline heave RMS、master displacement drift、memory force error。
- [ ] 生成 300 m RODM 实际算例验证报告 `docs/time_domain/mooring_validation_YYYY_MM_DD.md`。

## 6. 第六阶段：用户级工作流

- [x] 写一个最小可运行脚本：

```text
scripts/run_mooring_linear_matrix_demo.py
```

- [x] 脚本支持：
  - 一个节点线性系泊
  - 四角节点线性系泊
  - YAML 输入
  - 输出 reduced matrices
  - 输出简单响应指标
- [x] 在 `docs/mooring_framework_cn.md` 中加入完整运行命令。
- [x] 在 `docs/user_guide_cn.md` 或主 README 中加入系泊入口索引。
- [x] 在 `scripts/README.md` 中加入系泊 demo 说明。
- [x] 给用户提供推荐检查清单：
  - 节点编号
  - DOF 顺序
  - 矩阵单位
  - 是否删除 yaw
  - 频域是否启用系泊
  - 时域是否启用 pretension

## 7. 第七阶段：结果与报告输出

- [x] 在 WEC-Sim-like 平台 report 中加入 mooring 小节。
- [x] 输出 `mooring_summary.json`。
- [x] 输出 `mooring_reduced_stiffness.npy`。
- [x] 输出 `mooring_reduced_damping.npy`。
- [x] 输出 `mooring_reduced_pretension.npy`。
- [ ] 可选输出 global retained DOF 下的 `K/C/F0`。
- [ ] 对 large matrix 输出采用 `.npz` 或稀疏格式，避免结果目录膨胀。
- [x] 在报告中明确：
  - 数值结果是否因系泊改变。
  - 改变来自 stiffness、damping 还是 pretension。
  - 系泊模型是否只是线性化近似。

## 8. 第八阶段：更复杂的线性模型

- [ ] 支持一个 attachment 的局部坐标变换矩阵。
- [ ] 支持 fairlead 偏心点到节点参考点的 6DOF wrench 转换。
- [ ] 支持多条线自动合成一个 attachment 矩阵。
- [ ] 支持从外部线性化结果读取 `K/C/F0`。
- [ ] 支持 sparse matrix 装配，避免大模型 dense global matrix 成本过高。
- [ ] 支持对 `K_moor` 做对称化选项。
- [ ] 支持检查 `K_moor` 半正定性。
- [ ] 支持检查 `C_moor` 半正定性。

## 9. 第九阶段：Lookup Table 系泊

- [ ] 设计 `LookupTableMooring` 接口。
- [ ] 支持 1D/2D/6D 查表输入格式调研。
- [ ] 支持位移到力的插值。
- [ ] 支持对 lookup table 在平衡点附近线性化，输出 `K/C/F0`。
- [ ] 与 WEC-Sim lookup table 方法对齐命名和文档说明。
- [ ] 添加小维度查表单元测试。
- [ ] 添加一个用户 demo。

## 10. 第十阶段：MoorDyn / 外部系泊适配预留

- [ ] 调研 MoorDyn 输入文件、输出力和线性化接口。
- [ ] 设计 `ExternalMooringProvider` 抽象接口。
- [ ] 支持“外部程序先离线线性化，仓库读取 `K/C/F0`”的轻量路径。
- [ ] 设计实时耦合需要的 time-step force callback，但暂不作为默认实现。
- [ ] 明确 MoorDyn 耦合的坐标系、单位和 fairlead 点映射。
- [ ] 编写 `docs/moordyn_adapter_plan_cn.md`。

## 11. 第十一阶段：优化与设计变量

- [ ] 将系泊刚度纳入 `optimization` 包的设计变量。
- [ ] 支持每个 attachment 的 `surge/sway/heave` stiffness 扫描。
- [ ] 支持多角点 stiffness group。
- [ ] 支持约束：
  - 最大位移
  - 最大倾角
  - 最大系泊力
  - heave RMS 不劣化阈值
- [ ] 把系泊设计变量与 hinge stiffness / connector design 联合评估。
- [ ] 输出 Pareto 表和图件。

## 12. 第十二阶段：验收标准

完成整个 mooring 板块，至少应满足以下条件：

- [ ] 无系泊默认路径和历史结果一致。
- [x] 线性 `K/C/F0` 模型可由 Python API 使用。
- [x] 线性 `K/C/F0` 模型可由 YAML 配置使用。
- [x] 可接入主 time-domain runner。
- [x] 可接入 WEC-Sim-like 时域平台。
- [ ] 可选接入频域 RAO 求解。
- [x] 有 1DOF/2DOF 解析测试。
- [ ] 有 300 m RODM 实际算例验证。
- [x] 有完整用户文档。
- [x] 有基础验证报告、图片和 metrics 输出。
- [x] 有清晰声明：当前是线性化系泊，不是完整非线性线缆动力学。

## 13. 推荐执行顺序

建议按下面顺序推进：

1. 配置驱动：`mooring/config.py` + YAML 模板。已完成。
2. 时域主入口接入：`run_rodm_case_from_config.py --domain time`。已完成。
3. 结果输出：metrics、reduced matrix、报告。基础框架已完成。
4. 1DOF/2DOF 解析验证。已完成。
5. 300 m RODM 有/无系泊对比。
6. 频域 `K/C` 可选接入。
7. 用户 demo 和 README 索引。
8. Lookup table 设计。
9. MoorDyn/外部适配设计。
10. 优化模块接入。

## 14. 当前最近一步建议

下一步最适合做：

```text
使用真实 300 m RODM 数据做有/无系泊对比，并生成 `docs/time_domain/mooring_validation_YYYY_MM_DD.md`。
```

原因：

- Python API、YAML 配置、主时域入口和基础解析验证已经完成。
- 真实 300 m 算例可以检查系泊对 centerline heave、master drift 和 radiation state-space 对比的实际影响。
- 这一步需要外部 DM-FEM2D 水动力和结构矩阵数据，因此应作为基础框架后的第一个真实算例验收。
