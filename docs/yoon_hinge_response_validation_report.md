# Yoon 铰接模型位移响应对比验证报告

生成时间：2026-04-28 16:34:20

## 1. 验证目的

本轮验证面向单铰接和双铰接浮体模型，在 180 deg 波浪入射条件下，对中心线竖向位移响应进行对比。
脚本使用当前标准化的 RODM 铰接模块生成响应，并与本地保存的 Yoon et al. 数值曲线及实验点进行对比。

预期数值变化：本轮只新增铰接验证脚本、绘图、报告和一个元素刚度矩阵读取工具；没有修改频域求解、SEREP、铰接刚度组装等数值算法。

## 2. 重要数据状态

严格复现 notebook 中的 Yoon 专用模型目前还缺少以下原始输入文件：
- `E:\phd\Code\DM-FEM2D\StructureData\Yoon_hinge\Job_hinge_study_100_60_YoonModel_MASS1.mtx`
- `E:\phd\Code\DM-FEM2D\StructureData\Yoon_hinge\Job_hinge_study_100_60_YoonModel_STIF1.mtx`
- `E:\phd\Code\DM-FEM2D\HydrodynamicData\Yoon_hinge\DM10_direction180_slender180_rho1025.nc`

因此，本报告包含两类证据：
- 当前 793 节点 RODM 模型的单铰、双铰代理验证，可重复运行并可作为后续标准程序入口。
- 本地已有的历史 Yoon/RODM 对比 PDF 渲染图，用于追踪此前基于 Yoon 专用输入完成的对比结果。

## 3. 当前代理模型输入

- 质量矩阵：`E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_MASS1.mtx`
- 刚度矩阵：`E:\phd\Code\DM-FEM2D\StructureData\JobMesh5_5_STIF1.mtx`
- 元素刚度：`E:\phd\Code\DM-FEM2D\StructureData\ELEMENTSTIFFNESS_793.mtx`
- 180 deg 水动力文件：`E:\phd\Code\DM-FEM2D\HydrodynamicData\Yoga\BM10_145_direaction180.nc`
- 铰接刚度惩罚参数：`1.000e+16`

## 4. 单铰接对比

- 图像：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\figures\single_hinge_180deg_centerline_comparison.png`
- 响应文件：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\single_hinge_current_rodm_response.npy`
- 计算耗时：`9.85787` s

| 对比对象 | RMSE |
| --- | ---: |
| yoon_numerical | `0.63372` |
| experiment | `0.757343` |

## 5. 双铰接对比

- 图像：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\figures\double_hinge_180deg_centerline_comparison.png`
- 响应文件：`D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\double_hinge_current_rodm_response.npy`
- 计算耗时：`7.68277` s

| 对比对象 | RMSE |
| --- | ---: |
| yoon_0_1 | `0.669462` |
| yoon_0_2 | `0.662107` |
| yoon_0_3 | `0.668325` |
| experiment | `0.789196` |

## 6. 历史 Yoon/RODM 图件

| PDF | 渲染 PNG |
| --- | --- |
| `E:\OneDrive - sjtu.edu.cn\A_Work_done\RODM_AD\Hige\Yoon-1-hige-180.pdf` | `D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\pdf_renders\Yoon-1-hige-180.png` |
| `E:\OneDrive - sjtu.edu.cn\A_Work_done\RODM_AD\Hige\Yoon-2-hige-180-180-1.pdf` | `D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\pdf_renders\Yoon-2-hige-180-180-1.png` |
| `E:\OneDrive - sjtu.edu.cn\A_Work_done\RODM_AD\Hige\Yoon-2-hige-180-180-2.pdf` | `D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\pdf_renders\Yoon-2-hige-180-180-2.png` |
| `E:\OneDrive - sjtu.edu.cn\A_Work_done\RODM_AD\Hige\Yoon-2-hige-180-180-3.pdf` | `D:\OneDrive - 宁波东方理工大学（暂名）\Code_RODM\RODM_20250310\results\hinge_response_validation\pdf_renders\Yoon-2-hige-180-180-3.png` |

## 7. 判断

铰接刚度组装内核此前已经通过 legacy `DM_Hinge` 等价性和 Abaqus 模态基准验证；本轮进一步确认当前标准化入口可以完成单铰、双铰 RODM 响应计算、曲线对比和报告输出。
但由于 Yoon 专用质量矩阵、刚度矩阵和水动力 NetCDF 文件当前不在本地可访问路径中，严格的 Yoon 模型重算还不能判定为完成。
当前代理模型与 Yoon 曲线存在差异是预期结果，主要原因是结构模型、水动力模型和铰接位置定义不完全相同。

下一步应优先恢复 `StructureData/Yoon_hinge` 与 `HydrodynamicData/Yoon_hinge` 原始输入；恢复后可将本脚本中的代理模型分支替换为严格 Yoon 模型分支，重新生成同一套图和指标。
