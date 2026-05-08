# scripts 入口说明

本目录存放可重复运行的命令行入口。后续一体化软件建设中，建议把 notebook 中稳定的流程逐步迁移为这里的脚本，再沉淀到 `src/offshore_energy_sim/` 包。

## 推荐主入口

| 脚本 | 用途 |
| --- | --- |
| `check_environment.py` | 检查当前 Python 环境是否具备核心依赖。 |
| `run_regular_wave_batch_validation.py` | 连续性浮体 60/120/180/240/300 m 波长验证。 |
| `run_yoon_hinge_cases.py` | Yoon 单铰接、双铰接标准验证。 |
| `run_complex_hinge_10x10.py` | 10x10 模块铰接水弹性计算或输入检查。 |
| `build_hydroelastic_validation_report.py` | 汇总连续体和铰接验证结果，生成中文报告。 |
| `run_rodm_case_from_config.py` | 从 YAML 配置运行单个 RODM 频域算例。 |

## 验证/回归脚本

| 脚本 | 用途 |
| --- | --- |
| `validate_reduction_solver_kernels.py` | 检查 DOF 缩减、SEREP 辅助函数和频域求解核。 |
| `validate_structure_connectors.py` | 检查结构装配、连接件和铰接核函数。 |
| `validate_complex_hinge_10x10_setup.py` | 检查 10x10 网格、主节点和铰接节点配对。 |
| `validate_published_hinge_kernels.py` | 对照已发表铰接程序中的核函数和节点规则。 |
| `validate_environment_load_power_strength.py` | 检查环境、风载、响应谱、强度和光伏辅助函数。 |
| `run_refactor_regression_suite.py` | 聚合运行多项回归检查。 |

## 历史/专项诊断脚本

| 脚本 | 用途 |
| --- | --- |
| `plot_reference_case_300.py` | 300 m 连续体参考算例图件。 |
| `plot_reference_case_300_solver_variants.py` | 300 m 默认节点顺序与水动力节点反序候选对比。 |
| `investigate_reference_case_300_variants.py` | 300 m 历史变体诊断。 |
| `run_reference_case_300_workflow.py` | 300 m 参考算例工作流复现。 |
| `run_reference_case_300_rodm_compare.py` | 300 m RODM 历史/当前结果比较。 |
| `run_yoon_hinge_response_validation.py` | 早期 Yoon 铰接响应验证入口，当前优先使用 `run_yoon_hinge_cases.py`。 |
| `run_hinge_abaqus_benchmark.py` | Abaqus 铰接基准重跑入口，需要本机 Abaqus 和外部输入数据。 |

## 约定

- 脚本默认从项目根目录运行。
- 脚本内部会把 `src/` 加入 `PYTHONPATH`。
- 结果统一写入 `results/`，报告或说明性副本写入 `docs/`。
- 不建议新增只在本机有效的绝对路径；需要外部数据时优先使用参数或环境变量。
