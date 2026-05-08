# 300 m 波长连续性浮体对比偏差诊断

日期：2026-04-30

本文档专门说明连续性浮体 300 m 波长对比结果为什么容易显得“不准确”。结论先行：300 m 的问题不是一个单纯的画图反向问题，而是同时包含水动力节点顺序、历史基准来源、参考数据缺失和实验散点差异几个因素。

## 1. 当前看到的现象

300 m 当前至少有三类图：

| 图件 | 内容 | 用途 |
| --- | --- | --- |
| `results/regular_wave_batch/wavelength_300m/figures/regular_wave_300m_heave_comparison.png` | 默认 RODM/DM_Method 曲线 + 实验点 + Fu 曲线 | 早期默认节点顺序对比图。 |
| `results/regular_wave_batch/wavelength_300m/figures/regular_wave_300m_heave_selected.png` | hydro-node-reversed 曲线 + saved baseline + 默认节点顺序 | 当前方向/顺序修正诊断图。 |
| `figures/reference_case_300_solver_variants.png` | saved baseline + 默认解 + hydro-reversed 解 + 实验点 + Fu 曲线 | 最完整的 300 m 诊断图。 |

其中 `regular_wave_300m_heave_comparison.png` 的默认蓝线明显偏低；`regular_wave_300m_heave_selected.png` 中绿色线与黑色历史基准几乎重合，但没有叠加实验/Fu 曲线，因为 Mac 本地缺少 `exp_300.txt` 和 `fu_sim300.txt`。

## 2. 数值证据

以历史保存基准 `displacement_55mesh_300.npy` 为参照：

| 候选结果 | 与历史基准的中心线 heave RMSE | 相关系数 | 判断 |
| --- | ---: | ---: | --- |
| 默认节点顺序 | 0.0892402612 | 0.9751120291 | 明显不是历史论文基准。 |
| hydro-node-reversed | 0.0010529883 | 0.9999931819 | 几乎复现历史论文基准。 |

因此，300 m 默认曲线不准确的主因是：**水动力 10 个节点块的顺序与结构主节点顺序不一致**。修正方式不是把横坐标反过来，而是对水动力矩阵和波浪力按节点块反序：

```text
[node1, node2, ..., node10] -> [node10, node9, ..., node1]
```

局部 5 个 DOF 的顺序保持不变。

## 3. 为什么修正后仍与实验点不完全一致

hydro-node-reversed 结果只是复现历史保存基准，它并不保证完全穿过所有实验点。历史基准与外部曲线的参考误差为：

| 对比对象 | RMSE |
| --- | ---: |
| saved baseline vs `exp_300.txt` | 0.0636748225 |
| saved baseline vs `fu_sim300.txt` | 0.0448893490 |
| hydro-node-reversed vs `exp_300.txt` | 0.0639974545 |
| hydro-node-reversed vs `fu_sim300.txt` | 0.0448861941 |

也就是说，反序修正后已经回到历史基准水平；剩余偏差主要来自历史模型与实验/他人数值曲线之间本身的差异。

可能因素包括：

- `exp_300.txt` 只有 9 个实验点，且中部散点有明显离散。
- `fu_sim300.txt` 与 RODM 的结构离散、边界/水动力设置、阻尼或静水恢复处理可能不完全一致。
- 当前 Mac 本地没有 `exp_300.txt`、`fu_sim300.txt`、`DM10_300_direction0.nc` 和连续体结构矩阵原始输入，因此本机不能完整重新生成带实验/Fu 的 300 m 修正图，只能复用历史图件和响应数组。

## 4. 当前代码中的处理方式

批处理脚本 `scripts/run_regular_wave_batch_validation.py` 对 300 m 做了显式设置：

```python
HYDRO_NODE_REVERSE_BY_WAVELENGTH = {300: True}
```

这表示：

- 60 m、120 m、180 m、240 m 使用默认水动力节点顺序。
- 300 m 使用 `hydro_reversed` 响应作为当前选定结果。

该设置符合已有诊断结果：300 m 反序解与历史保存基准几乎重合。

## 5. 当前不足

现在报告里容易产生误解的地方是：300 m 有一张旧默认对比图和一张方向修正图。

- 旧默认对比图包含实验/Fu，但默认 RODM 曲线不是最终推荐曲线。
- 方向修正图包含推荐 RODM 曲线，但由于本地缺少实验/Fu 文本数据，没有重新叠加实验/Fu。

所以如果只看其中一张，会觉得“不准确”或“对比不完整”。

## 6. 建议修正

短期建议：

1. 把 `exp_300.txt`、`fu_sim300.txt`、`DM10_300_direction0.nc`、`JobMesh5_5_MASS1.mtx`、`JobMesh5_5_STIF1.mtx` 同步到 `/Users/yongkang/data/DM-FEM2D` 的标准目录。
2. 重新运行：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_regular_wave_batch_validation.py
```

3. 生成一张新的 300 m 最终图：`hydro-node-reversed RODM + Experiment + Fu et al.`。

中期建议：

- 将 300 m 旧默认图标记为“历史默认节点顺序，不作为最终对比图”。
- 总报告中优先展示 `figures/reference_case_300_solver_variants.png` 或重新生成的 hydro-reversed 外部对比图。
- 保留默认解与 hydro-reversed 解的差异图，作为节点顺序溯源证据。

## 7. 结论

300 m 对比不准确的直接原因是默认水动力节点顺序与结构主节点顺序不一致；这个问题已通过 `hydro-node-reversed` 修正。修正后仍与实验点存在偏差，这是历史 RODM 基准本身与实验/他人数值数据之间的剩余差异，而不是本轮 Mac 重构造成的新错误。
