# 线性系泊框架验证报告

日期：2026-05-24 16:06:45

本报告验证当前 `offshore_energy_sim.mooring` 线性系泊框架。验证不依赖外部 DM-FEM2D 数据，所有算例均为小矩阵解析或半解析检查。

## 验证结论

- 总体结果：`passed`
- 检查数量：`6`
- metrics：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\metrics.json`

## 图件

- 误差汇总图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\linear_mooring_error_summary.png`

## 检查项

| 检查 | 是否通过 | 关键误差 |
| --- | --- | ---: |
| `force_formula` | `True` | `0.000000e+00` |
| `nodal_assembly_6dof_to_5dof` | `True` | `0.000000e+00` |
| `reduced_projection` | `True` | `0.000000e+00` |
| `config_provider` | `True` | `0.000000e+00` |
| `sdof_frequency_time_closure` | `True` | `1.489921e-04` |
| `coupled_2dof_frequency_time_closure` | `True` | `2.360569e-05` |

## 1DOF 时域/频域闭合

- 谐波复幅值相对误差：`1.489921e-04`
- 静态偏置解析值：`1.000000e-01`
- 静态偏置时域估计：`1.000590e-01`
- 静态偏置绝对误差：`5.904213e-05`
- 时序对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\sdof_time_frequency_comparison.png`
- 复幅值对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\sdof_complex_amplitude_comparison.png`

## 其他检查图件

- 公式对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\force_formula_comparison.png`
- 节点装配误差图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\nodal_assembly_error.png`
- 降阶投影对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\reduced_projection_comparison.png`
- 配置 provider 对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\config_provider_comparison.png`
- 2DOF 耦合复幅值对比图：`C:\Users\WYJ\Documents\Codex\2026-05-09\macbook-github-windows-macbook-git-clone\yk202605\results\mooring\linear_framework_validation\figures\coupled_2dof_complex_amplitude_comparison.png`

## 数值影响

本验证只新增验证脚本、结果和报告，不改变 RODM 频域核心。线性系泊项只有在用户显式传入 `K_moor/C_moor/F0` 时才会改变响应。
