# 时域模拟研究资料

本文件夹用于存放从当前频域 RODM 平台推进到时域模拟的调研、设计和后续实现记录。

当前文档：

- `time_domain_research_and_implementation_plan.md`：本地代码调研、Capytaine/WEC-Sim 方法调研、推荐方程、模块设计、验证路线和阶段计划。
- `time_domain_implementation_status.md`：当前已实现的 Python 模块、验证脚本、测试结果和下一步数据验证方式。
- `basic_case_time_series_guide.md`：300 m 基础 RODM 算例时序计算的运行步骤和输出说明。

建议后续继续在本文件夹中追加：

- `equation_conventions.md`：时域方程符号、相位约定、DOF 顺序和单位约定。
- `validation_cases.md`：1DOF、规则波稳态、随机波 RMS、WEC-Sim 对照等验证算例记录。
- `implementation_notes.md`：实际编码过程中的数值稳定性、性能和接口决策。
