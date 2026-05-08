# 标准化代码逐行备注说明

日期：2026-04-30

本文档是源码旁路注释，不直接把每一行源代码都塞入中文注释，避免主程序变得难读。阅读方式是：先看“主流程逐行说明”，再按文件和函数跳转到源码。所有节点编号均沿用 Abaqus/旧 notebook 的一基编号，Python 数组索引仍是一基转零基后的内部实现。

## 1. 主流程逐行说明：Yoon 单铰/双铰验证

入口文件：[run_yoon_hinge_cases.py](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [run_yoon_hinge_cases.py:31](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:31) | `case_manifest(case)` | 把算例对象转成可写入 JSON 的清单，记录矩阵路径、铰接线、主节点、对比曲线等复现信息。 |
| [run_yoon_hinge_cases.py:73](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:73) | `render_legacy_figures(case, output_dir)` | 用 macOS Quick Look 把历史论文 PDF 图件转成 PNG，方便和当前 RODM 图拼接对照。 |
| [run_yoon_hinge_cases.py:109](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:109) | `compose_current_legacy_panels(...)` | 把当前计算图和历史论文图上下拼接，生成视觉对比图。 |
| [run_yoon_hinge_cases.py:156](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:156) | `write_report(...)` | 写中文运行报告，列出每个算例是否求解成功、图片位置、响应数组位置和缺失输入。 |
| [run_yoon_hinge_cases.py:222](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:222) | `parse_args()` | 解析命令行参数，例如 `--case all`、`--data-root`、`--output-root`。 |
| [run_yoon_hinge_cases.py:236](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_yoon_hinge_cases.py:236) | `main()` | 主控函数：构造算例、检查输入、求解、保存 `.npy`、绘图、渲染历史图、写报告。 |

核心计算文件：[yoon_hinge.py](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [yoon_hinge.py:38](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:38) | `ComparisonLineSpec` | 描述一条要画的对比曲线，例如中心线、上边线、下边线，以及是否反转模型 x 方向。 |
| [yoon_hinge.py:50](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:50) | `YoonHingeCase` | 单个 Yoon 铰接验证算例的完整输入容器，包括结构矩阵、水动力文件、铰接节点、主节点、降阶方法和后处理设置。 |
| [yoon_hinge.py:95](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:95) | `YoonHingeResult` | 单个算例求解后的结果容器，包含全局响应、heave 网格和角频率。 |
| [yoon_hinge.py:104](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:104) | `yoon_hinge_data_root()` | 确定外部数据根目录，优先读取环境变量 `RODM_DM_FEM_ROOT`，否则使用本机默认路径。 |
| [yoon_hinge.py:112](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:112) | `default_reference_root()` | 返回本地论文图件和数字化 CSV 的参考数据目录。 |
| [yoon_hinge.py:120](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:120) | `block_diagonal_repeat(matrix, count)` | 把单模块结构矩阵按块对角重复，用于单铰 2 模块、双铰 3 模块装配。 |
| [yoon_hinge.py:128](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:128) | `missing_input_paths(case)` | 检查质量矩阵、刚度矩阵、水动力文件是否存在。 |
| [yoon_hinge.py:134](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:134) | `_reverse_node_order_vector(...)` | 按节点块反转力或位移向量，兼容旧 notebook 的水动力节点排序。 |
| [yoon_hinge.py:144](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:144) | `_static_condensation_reduce(...)` | 静态凝聚实现，保留旧 notebook 的质量矩阵投影顺序选项。 |
| [yoon_hinge.py:177](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:177) | `_reduce_structural_matrices(...)` | 结构侧主流程：复制刚度、加入铰接、删除第 6 自由度、划分主从 DOF，再做 SEREP 或静态凝聚。 |
| [yoon_hinge.py:223](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:223) | `_read_hydrodynamic_terms(...)` | 读取 NetCDF 中的附加质量、辐射阻尼、静水刚度和波浪力，并删除不参与求解的 DOF。 |
| [yoon_hinge.py:271](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:271) | `solve_yoon_hinge_case(case)` | 单铰/双铰的完整求解函数：结构降阶、水动力降阶、频域方程求解、全局响应重构、heave 网格提取。 |
| [yoon_hinge.py:322](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:322) | `extract_yoon_hinge_heave_grid(...)` | 从全局 5DOF 响应中抽取第 3 个自由度 heave，并拼成论文对比用二维网格。 |
| [yoon_hinge.py:342](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:342) | `_load_xy(path)` | 读取两列参考曲线 CSV 或文本文件。 |
| [yoon_hinge.py:354](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:354) | `plot_yoon_hinge_case(...)` | 对每个配置的剖面画 RODM 曲线、Yoon 曲线和试验点。 |
| [yoon_hinge.py:412](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:412) | `_hinge_spec(...)` | 用简短参数生成 `ExplicitHingeSpec`，减少单铰/双铰节点配置重复。 |
| [yoon_hinge.py:431](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/yoon_hinge.py:431) | `build_yoon_hinge_cases(...)` | 构造全部标准 Yoon 算例：`single_180`、`double_180`、`double_210`、`double_240`、`double_270`。 |

## 2. 主流程逐行说明：10x10 模块铰接水弹性

入口文件：[run_complex_hinge_10x10.py](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_complex_hinge_10x10.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [run_complex_hinge_10x10.py:31](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_complex_hinge_10x10.py:31) | `case_manifest(case)` | 输出 10x10 算例清单：100 个模块、180 条铰接线、1260 对节点、主节点、数据路径和水动力设置。 |
| [run_complex_hinge_10x10.py:85](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_complex_hinge_10x10.py:85) | `write_report(...)` | 写中文报告，说明当前是否缺输入、旧程序约定、运行命令和输出位置。 |
| [run_complex_hinge_10x10.py:153](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_complex_hinge_10x10.py:153) | `parse_args()` | 解析 `--skip-solve`、`--data-root`、`--output-root` 等命令行参数。 |
| [run_complex_hinge_10x10.py:166](/Users/yongkang/Projects/RODM_20250310_local/scripts/run_complex_hinge_10x10.py:166) | `main()` | 主控函数：构造 10x10 case、检查输入、必要时求解、保存响应和图片。 |

核心计算文件：[complex_hinge_10x10.py](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [complex_hinge_10x10.py:48](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:48) | `ComplexHingeCase` | 10x10 算例输入容器，保存结构矩阵、水动力文件、模块网格、铰接线、主节点和旧程序兼容选项。 |
| [complex_hinge_10x10.py:85](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:85) | `ComplexHingeResult` | 10x10 求解结果容器，包括全局响应、70x70 原始 heave 网格、61x61 合并界面网格。 |
| [complex_hinge_10x10.py:95](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:95) | `complex_hinge_data_root()` | 返回外部数据根目录。 |
| [complex_hinge_10x10.py:103](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:103) | `build_complex_hinge_10x10_case(...)` | 从旧 notebook 约定构造 10x10 标准算例。默认刚度 `1e10`，释放 DOF 小惩罚 `10`。 |
| [complex_hinge_10x10.py:132](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:132) | `missing_input_paths(case)` | 检查 10x10 质量矩阵、刚度矩阵和水动力文件是否存在。 |
| [complex_hinge_10x10.py:138](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:138) | `_sparse_block_diagonal_repeat(...)` | 用稀疏矩阵把单模块结构矩阵重复 100 次，避免 29400x29400 稠密矩阵占用过大内存。 |
| [complex_hinge_10x10.py:148](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:148) | `_reduce_sparse_matrix_dofs(...)` | 在稀疏矩阵上删除每个节点的第 6 自由度。 |
| [complex_hinge_10x10.py:156](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:156) | `_reverse_node_order_vector(...)` | 提供按节点块反转向量的选项，兼容不同水动力节点排序。 |
| [complex_hinge_10x10.py:162](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:162) | `_static_condensation_sparse(...)` | 对 24500x24500 结构矩阵做稀疏静态凝聚，得到 500x500 主自由度矩阵。 |
| [complex_hinge_10x10.py:209](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:209) | `_read_hydrodynamic_terms(case)` | 读取 100-body 水动力数据，删除 yaw-like DOF，得到 500x500 水动力矩阵和 500 维波浪力。 |
| [complex_hinge_10x10.py:259](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:259) | `solve_complex_hinge_case(case)` | 完整 10x10 求解函数：结构稀疏装配、铰接矩阵加入、静态凝聚、水动力耦合、频域求解和响应重构。 |
| [complex_hinge_10x10.py:341](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:341) | `extract_complex_hinge_heave_grid(...)` | 从 4900 个结构节点响应中抽取 heave，并拼成 70x70 或 61x61 响应场。 |
| [complex_hinge_10x10.py:374](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/validation/complex_hinge_10x10.py:374) | `plot_complex_hinge_result(...)` | 输出 10x10 heave 热力图和中心线响应图。 |

## 3. 铰接与模块网格工具说明

模块网格文件：[modular_grid.py](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [modular_grid.py:20](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:20) | `ModuleGridSpec` | 定义模块阵列尺寸、单模块节点数、模块物理尺寸、每节点 DOF 数和中心节点编号。 |
| [modular_grid.py:35](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:35) | `module_count` | 返回模块总数，例如 10x10 为 100。 |
| [modular_grid.py:41](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:41) | `nodes_per_module` | 返回每个模块结构节点数，例如 7x7 为 49。 |
| [modular_grid.py:47](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:47) | `total_nodes` | 返回整体结构节点数，例如 100x49 为 4900。 |
| [modular_grid.py:53](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:53) | `structure_size` | 返回整体边长，例如 10x30 m 为 300 m。 |
| [modular_grid.py:60](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:60) | `ModuleControlPoint` | 记录一个模块中心主节点的编号、坐标和所在模块行列。 |
| [modular_grid.py:71](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:71) | `module_offset_one_based(...)` | 根据模块行列返回该模块在整体节点编号中的起始偏移。 |
| [modular_grid.py:82](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:82) | `generate_module_center_control_points(...)` | 生成 100 个模块中心控制点，顺序为从上到下、从左到右。 |
| [modular_grid.py:112](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:112) | `generate_master_nodes_one_based(...)` | 只返回控制点中的 FEM 节点编号，用于主自由度划分。 |
| [modular_grid.py:118](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:118) | `generate_x_hinge_node_pairs(...)` | 生成左右相邻模块的边界节点对，已验证与旧 `generate_hinge_x_pairs` 完全一致。 |
| [modular_grid.py:137](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:137) | `generate_y_hinge_node_pairs(...)` | 生成上下相邻模块的边界节点对，已验证与旧 `generate_hinge_y_pairs` 完全一致。 |
| [modular_grid.py:156](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:156) | `generate_grid_hinge_specs(...)` | 把 x/y 节点对转换成 `ExplicitHingeSpec`，自动设置 x 向释放 DOF4、y 向释放 DOF3。 |
| [modular_grid.py:198](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/modular_grid.py:198) | `drop_duplicate_module_interfaces(...)` | 删除模块拼接界面重复行列，将 70x70 响应场转换成连续 61x61 响应场。 |

铰接工具文件：[hinges.py](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [hinges.py:21](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:21) | `HingeLineSpec` | 按两列节点定义一条铰接线，适合规则网格列连接。 |
| [hinges.py:77](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:77) | `ExplicitHingeSpec` | 显式给出两侧节点列表，适合单铰、双铰、10x10 等复杂连接。 |
| [hinges.py:104](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:104) | `calculate_column_node_indices(...)` | 根据列号生成该列全部节点编号。 |
| [hinges.py:120](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:120) | `generate_column_elements(...)` | 根据两列节点生成跨铰接线的四节点单元列表。 |
| [hinges.py:139](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:139) | `remove_element_stiffness_in_place(...)` | 从全局刚度中扣除铰接线处原壳单元刚度，适合早期显式切缝建模。 |
| [hinges.py:162](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:162) | `hinge_coupling_matrix(...)` | 生成 6x6 铰接耦合矩阵 `KC`，释放自由度可设为 0、10、100 等小惩罚。 |
| [hinges.py:181](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:181) | `add_hinge_connections_in_place(...)` | 对每对节点加入 `+KC/-KC` 四个块，形成相对位移惩罚连接。 |
| [hinges.py:207](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:207) | `apply_hinge_line_in_place(...)` | 将 `HingeLineSpec` 应用到全局刚度矩阵。 |
| [hinges.py:229](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:229) | `apply_explicit_hinge_in_place(...)` | 将一个 `ExplicitHingeSpec` 应用到全局刚度矩阵。 |
| [hinges.py:246](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:246) | `apply_explicit_hinges_in_place(...)` | 批量应用多个显式铰接定义。 |
| [hinges.py:257](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:257) | `assemble_explicit_hinges_sparse(...)` | 10x10 使用的稀疏铰接装配，避免构造超大稠密铰接矩阵。 |
| [hinges.py:304](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:304) | `remove_hinge_line_elements_in_place(...)` | 对 `HingeLineSpec` 自动扣除跨铰接线单元刚度。 |
| [hinges.py:324](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:324) | `build_hinged_stiffness(...)` | 从原始刚度复制一份，完成切缝扣除和铰接连接加入。 |
| [hinges.py:340](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:340) | `read_symmetric_element_stiffness_matrix(...)` | 读取 Abaqus 输出的单元对称刚度矩阵。 |
| [hinges.py:385](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/structure/hinges.py:385) | `read_plain_upper_triangle_stiffness_matrix(...)` | 读取纯上三角格式的对称刚度矩阵。 |

## 4. 优化接口说明

文件：[connectors.py](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py)

| 位置 | 代码对象 | 说明 |
| --- | --- | --- |
| [connectors.py:26](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py:26) | `ConnectorDesignVariable` | 描述一个连接件优化变量，例如 x 向铰接刚度、y 向铰接刚度、释放刚度或开关变量。 |
| [connectors.py:37](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py:37) | `normalized_initial_value()` | 将初始值按尺度归一化，方便后续优化算法使用。 |
| [connectors.py:44](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py:44) | `ConnectorObjectiveSpec` | 描述优化目标，例如最大 heave、平均 heave、连接件力等。 |
| [connectors.py:53](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py:53) | `ConnectorOptimizationProblem` | 把算例 ID、设计变量、目标函数和约束组织成一个优化问题。 |
| [connectors.py:62](/Users/yongkang/Projects/RODM_20250310_local/src/offshore_energy_sim/optimization/connectors.py:62) | `uniform_hinge_stiffness_variables(...)` | 生成最基础的两个优化变量：`k_hinge_x` 和 `k_hinge_y`。 |

## 5. 推荐阅读顺序

1. 只想运行算例：先读 [user_guide_cn.md](/Users/yongkang/Projects/RODM_20250310_local/docs/user_guide_cn.md)。
2. 想理解铰接装配：读 `modular_grid.py` 和 `hinges.py` 两节。
3. 想理解 10x10 水弹性：读 `complex_hinge_10x10.py` 一节。
4. 想理解单铰/双铰验证：读 `yoon_hinge.py` 一节。
5. 想做优化：读 `optimization/connectors.py` 一节，再看 10x10 case 中 `k_hinge` 和 `released_dof_stiffness` 的传入方式。
