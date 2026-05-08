# 铰接模型验证与程序包化报告

生成时间：2026-04-28 15:02:08

## 1. 验证目标

本报告针对铰接模型完成两层验证：

- 旧脚本 `DM_Hinge.py` 与新程序包 `offshore_energy_sim.structure.hinges` 的矩阵装配核函数等价性；
- `Job-1_largemesh_hinge_1.inp` / `Job-1_largemesh_hinge.dat` 铰接算例的 Abaqus 模态频率复现与矩阵一致性检查。

预期数值变化：本轮只新增标准接口、验证脚本和报告，不修改原始数值算法；旧脚本应保持可运行。

## 2. 铰接程序包接口

新增标准入口位于 `src/offshore_energy_sim/structure/hinges.py`：

- `HingeLineSpec`：定义两侧节点列、每行节点数、列数、铰接刚度、释放自由度；
- `apply_hinge_line_in_place`：在全局刚度矩阵中加入铰接连接；
- `remove_hinge_line_elements_in_place`：可选移除铰接线两侧的壳单元刚度贡献；
- `build_hinged_stiffness`：面向后续优化调用的组合接口。

## 3. 输入数据

- 铰接 Abaqus 输入：`E:\phd\Code\DM-FEM2D\Fem_inp\Job-1_largemesh_hinge_1.inp`
- 铰接 Abaqus 输出：`E:\phd\Code\DM-FEM2D\Fem_inp\Job-1_largemesh_hinge.dat`
- 63 节点刚度矩阵：`E:\phd\Code\DM-FEM2D\StructureData\Job-1_largemesh_STIF1.mtx`
- 63 节点质量矩阵：`E:\phd\Code\DM-FEM2D\StructureData\Job-1_largemesh_ConsistentMass_MASS1.mtx`

## 4. 结构与边界信息

- 节点数：`63`
- 壳单元数：`40`
- 约束自由度数：`45`
- 矩阵维度：`(378, 378)`

## 5. 旧脚本等价性

- 节点配对数：`3`
- 最大绝对误差：`0`
- L2 误差：`0`

## 6. Abaqus dat 溯源

- 是否发现重运行 dat：`True`
- 重运行 dat：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_validation\abaqus_work\Job-1_largemesh_hinge_1.dat`
- 原始 dat 与重运行 dat 的特征值最大相对误差：`1.7203988891694015`
- 原始 dat 与重运行 dat 的圆频率最大相对误差：`0.6494001747835068`

说明：`Job-1_largemesh_hinge_1.inp` 重运行结果与历史 `Job-1_largemesh_hinge.dat` 不一致，
因此历史 dat 很可能来自另一个缺失的 `Job-1_largemesh_hinge.inp` 或不同边界设置。当前程序包验证采用可复现的 `_1.inp` 重运行 dat。

## 7. 程序包矩阵验证

- 参考 dat：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_validation\abaqus_work\Job-1_largemesh_hinge_1.dat`
- 对比模态阶数：`20`
- 特征值 RMSE：`0.0137112`
- 特征值最大相对误差：`3.85933e-05`
- 圆频率最大相对误差：`3.68203e-05`
- 对比图：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_validation\figures\hinge_modal_frequency_comparison.png`

该项使用现有 63 节点矩阵文件和 `_1.inp` 重运行 Abaqus dat 验证边界约束/矩阵求解流程。

| 模态 | Abaqus 特征值 | 程序包特征值 | 相对误差 |
| ---: | ---: | ---: | ---: |
| 1 | `0.51084` | `0.510841` | `1.84563e-06` |
| 2 | `4.3102` | `4.31021` | `1.49367e-06` |
| 3 | `12.161` | `12.1609` | `5.43222e-06` |
| 4 | `20.963` | `20.963` | `8.54493e-07` |
| 5 | `28.613` | `28.6129` | `5.00761e-06` |
| 6 | `106.46` | `106.461` | `9.96167e-06` |
| 7 | `112.37` | `112.374` | `3.83948e-05` |
| 8 | `179.47` | `179.466` | `2.21137e-05` |
| 9 | `219.41` | `219.414` | `1.97664e-05` |
| 10 | `259.25` | `259.251` | `2.89139e-06` |
| 11 | `274.36` | `274.356` | `1.61928e-05` |
| 12 | `431.59` | `431.594` | `9.06011e-06` |
| 13 | `655.26` | `655.264` | `5.97142e-06` |
| 14 | `793.12` | `793.122` | `2.97749e-06` |
| 15 | `947.32` | `947.316` | `4.24822e-06` |
| 16 | `985.09` | `985.089` | `1.39139e-06` |
| 17 | `1058.8` | `1058.82` | `1.95355e-05` |
| 18 | `1260.9` | `1260.85` | `3.85933e-05` |
| 19 | `1352.6` | `1352.63` | `1.90094e-05` |
| 20 | `1396.8` | `1396.79` | `9.44574e-06` |

## 8. 结论

铰接连接的标准程序包接口已形成，可直接被后续优化、批量参数分析和一体化平台调用。
旧脚本与新接口的铰接矩阵装配已经达到零误差；现有 63 节点矩阵与 `_1.inp` 重运行 Abaqus dat 的模态对比已经通过。
历史 `Job-1_largemesh_hinge.dat` 与 `_1.inp` 不一致，应作为待溯源数据保留，不应作为当前 `_1` 算例的通过/失败标准。
