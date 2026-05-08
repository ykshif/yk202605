# RODM 水动力计算窗口用户说明

日期：2026-05-02

本文档说明如何使用本地水动力计算窗口生成 Capytaine 风格 `.nc` 文件，并把结果传给 RODM 后续频域水弹性模型。当前窗口只负责水动力部分：生成多浮体阵列几何，调用 Capytaine 计算辐射/绕射水动力，写出 NetCDF，并计算 RAO 用于界面运动预览。

## 1. 打开计算窗口

在项目根目录运行：

```bash
cd /Users/yongkang/Projects/RODM_20250310_local
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_hydrodynamics_ui.py --host 127.0.0.1 --port 8765
```

终端出现下面信息后，浏览器打开：

```text
RODM hydrodynamics UI: http://localhost:8765/
```

访问：

```text
http://localhost:8765/
```

停止窗口服务时，在启动服务的终端按 `Ctrl+C`。

如果 `8765` 端口已经被占用，脚本会自动尝试后续端口。以终端打印的 URL 为准。

## 2. 界面分区

窗口分为两部分：

| 区域 | 作用 |
| --- | --- |
| 左侧参数区 | 输入浮体尺寸、阵列数量、水深、频率、输出路径等。 |
| 左侧命令框 | 自动生成本次计算的 JSON 输入；也可以直接编辑 JSON 后点击计算。 |
| 右侧运动预览 | 显示浮体阵列在波浪上的运动。计算前是参数预览，计算后会使用 RAO 复幅值驱动运动。 |
| 右侧日志区 | 显示后台任务状态、Capytaine 求解进度和 `.nc` 输出路径。 |

计算完成后，右上角 `RAO` 状态会从“待算”变成“已算”，右侧运动模式会从“参数预览”变成“RAO 驱动”。

## 3. 浮体模块参数

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| 长度 | m | 单个矩形浮体在 x 方向的长度。 |
| 宽度 | m | 单个矩形浮体在 y 方向的宽度。 |
| 高度 | m | 单个浮体总高度。 |
| 吃水 | m | 静水面以下的浸没深度。程序按 `center_z = height / 2 - draft` 放置浮体。 |
| 水平网格 | m | Capytaine 几何面元水平尺度。数值越小，网格越细，计算越慢。 |
| 竖向网格 | m | Capytaine 几何面元竖向尺度。旧 notebook 常用 `0.2`。 |
| 质量 | kg | 单个浮体质量。留空时按 `rho * length * width * draft` 自动计算。 |

重要：如果想复现历史 Yoon/10x10 数据，建议显式填写质量，不要留空。旧 `.nc` 中有些文件的 `rho` 坐标和静水力/惯性矩阵使用的历史质量约定并不完全一致。

## 4. 阵列布局参数

| 字段 | 说明 |
| --- | --- |
| 行数 | y 方向模块数量。 |
| 列数 | x 方向模块数量。 |
| X 间距 | 相邻模块中心点 x 方向距离。 |
| Y 间距 | 相邻模块中心点 y 方向距离。 |

DOF 命名规则为：

```text
0_0__Surge, 0_0__Sway, 0_0__Heave, 0_0__Roll, 0_0__Pitch, 0_0__Yaw,
1_0__Surge, ...
```

这与当前 RODM 旧水动力 `.nc` 数据一致。后续 RODM 频域求解会继续按每个浮体 6 DOF 读取，并在结构降阶阶段删除 yaw/rz。

## 5. 海况与求解参数

| 字段 | 说明 |
| --- | --- |
| 水深 | 有限水深时填写正数，例如 `58.5`。 |
| 无限水深 | 勾选后忽略水深输入，Capytaine 使用无限水深。 |
| 密度 | 水密度 `rho`，常用 `1025` 或历史验证中的 `1000`。 |
| 重力 | 默认 `9.81`。 |
| 并行核数 | Capytaine `solve_all(..., n_jobs=...)` 的并行数量。默认建议 `1`；如果环境安装了 `joblib`，可设为更大的整数。 |
| 波浪方向 | 角度制，单位 deg。多个方向可写成 `0, 90, 180`。 |

波浪方向会在后台转换为弧度传给 Capytaine。

## 6. 频率设置

窗口使用角频率 `omega`，单位 `rad/s`。

| 模式 | 说明 |
| --- | --- |
| 单频 | 只计算一个 `omega`。适合当前 RODM 单频验证。 |
| 范围 | 输入起点、终点、数量，程序用 `linspace` 生成频率数组。 |
| 列表 | 手动输入多个频率，例如 `0.4, 0.5851, 0.8`。 |

历史 180 m 波长水动力文件常用：

```text
omega = 0.5851 rad/s
```

当前窗口不直接输入波长。如果只有波长，需要先换算成角频率；深水近似可用 `omega = sqrt(2*pi*g / wavelength)`，有限水深应使用色散关系求解。

## 7. 输出文件命名和保存位置

输出路径在“NetCDF 文件”输入框中设置。可以使用相对路径或绝对路径。

推荐使用相对路径，保存到项目结果目录：

```text
results/hydrodynamics_ui/array_hydrodynamics.nc
```

相对路径会自动解释为：

```text
/Users/yongkang/Projects/RODM_20250310_local/results/hydrodynamics_ui/array_hydrodynamics.nc
```

建议命名包含关键信息：

```text
results/hydrodynamics_ui/DM10x10_L30_W30_D1p1_omega0p5851_dir0.nc
results/hydrodynamics_ui/DM10_slender_L30_W60_D1p1_omega0p5851_dir180.nc
results/hydrodynamics_ui/test_2x2_coarse_omega0p5.nc
```

命名建议：

| 建议 | 原因 |
| --- | --- |
| 使用英文、数字、下划线 | 方便脚本读取和跨平台同步。 |
| 把模块数、尺寸、吃水、频率、方向写进文件名 | 后续不容易混淆不同水动力文件。 |
| 不要覆盖历史基准 `.nc` | 旧文件用于验证和溯源。 |
| 大算例先用 `_coarse` 或 `_trial` 命名 | 方便区分试算和正式计算。 |

如果输出目录不存在，程序会自动创建。

## 8. 命令框如何使用

左侧“命令框”显示本次计算的完整 JSON 输入。普通使用时只需要改上方表单即可，命令框会自动同步。

也可以直接编辑命令框，例如批量修改：

```json
{
  "layout": {
    "rows": 10,
    "columns": 10,
    "spacing_x_m": 30.01,
    "spacing_y_m": 30.01
  }
}
```

注意：

- 点击“计算 .nc”时，后台使用命令框中的 JSON，而不是重新读表单。
- 如果编辑命令框后又修改上方任意表单字段，命令框会被重新生成。
- 手动编辑 JSON 时必须保持合法格式，不能多逗号或漏括号。

## 9. 一键计算流程

1. 填写浮体模块参数。
2. 填写阵列行列数和间距。
3. 设置水深、密度、波浪方向和频率。
4. 设置输出 `.nc` 路径。
5. 点击“计算 .nc”。
6. 在右侧日志区等待 `Hydrodynamic NetCDF generation completed`。
7. 使用输出路径中的 `.nc` 文件进入 RODM 后续计算。

后台会执行：

```text
建立矩形浮体阵列
生成 Capytaine RadiationProblem 和 DiffractionProblem
调用 BEMSolver.solve_all
assemble_dataset
计算 RAO
separate_complex_values
写入 .nc
```

输出 `.nc` 主要变量包括：

| 变量 | 说明 |
| --- | --- |
| `added_mass` | 附加质量矩阵。 |
| `radiation_damping` | 辐射阻尼矩阵。 |
| `diffraction_force` | 绕射力。 |
| `Froude_Krylov_force` | Froude-Krylov 力。 |
| `excitation_force` | 激励力，等于上面两项之和。 |
| `inertia_matrix` | 刚体惯性矩阵。 |
| `hydrostatic_stiffness` | 静水恢复刚度。 |
| `rao` | 响应幅值算子，用于运动预览和检查。 |

## 10. 计算量估算

Capytaine BEM 问题数量大致为：

```text
问题数 = 6 * 浮体数量 * 频率数量 + 频率数量 * 波浪方向数量
```

例子：

| 算例 | 问题数 |
| --- | ---: |
| 1 个浮体，1 个频率，1 个方向 | 7 |
| 10 个浮体，1 个频率，1 个方向 | 61 |
| 100 个浮体，1 个频率，1 个方向 | 601 |
| 100 个浮体，40 个频率，1 个方向 | 24040 |

正式 10x10 阵列建议先用单频、较粗网格试算，确认输出路径、DOF 命名和日志正常，再加密网格或增加频率范围。

## 11. 常用参数模板

### 11.1 10x10 方形模块，180 m 波长单频

用于接近旧 `DM10_10_direction0_wl180.nc` 的设置：

| 参数 | 值 |
| --- | --- |
| 长度/宽度/高度 | `30 / 30 / 4` |
| 吃水 | `1.1` |
| 质量 | `990000` |
| 水平网格 | `4` |
| 竖向网格 | `0.2` |
| 行数/列数 | `10 / 10` |
| X/Y 间距 | `30.01 / 30.01` |
| 水深 | 勾选无限水深 |
| 密度 | 旧静水力验证用 `1000`；物理海水计算常用 `1025` |
| omega | `0.5851` |
| 波浪方向 | `0` |

输出命名示例：

```text
results/hydrodynamics_ui/DM10x10_L30_W30_D1p1_omega0p5851_dir0.nc
```

### 11.2 DM10 slender 单模块验证

用于和旧 `DM10_direction180_slender180_rho1025.nc` 的首模块静水力/惯性块对比：

| 参数 | 值 |
| --- | --- |
| 长度/宽度/高度 | `30 / 60 / 4` |
| 吃水 | `1.1` |
| 质量 | `2029500` |
| 水平网格 | `2` |
| 竖向网格 | `0.2` |
| 行数/列数 | `1 / 1` |
| X/Y 间距 | `30.01 / 60.01` |
| 水深 | 无限水深 |
| 密度 | `1000` |
| omega | `0.5851` |
| 波浪方向 | `180` |

输出命名示例：

```text
results/hydrodynamics_ui/DM10_slender_L30_W60_D1p1_omega0p5851_dir180.nc
```

## 12. 对比验证

启动 UI 后，可以运行：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/validate_hydrodynamics_ui_against_nc.py --ui-url http://localhost:8765
```

输出目录：

```text
results/hydrodynamics_ui_validation
```

主要报告：

```text
results/hydrodynamics_ui_validation/hydrodynamics_ui_validation_report.md
```

该脚本会：

1. 读取已有 Yoon/10x10 `.nc` 文件元信息。
2. 通过 UI API 重新生成小型验证 `.nc`。
3. 对比首模块 DOF 命名、惯性矩阵、静水恢复刚度。
4. 检查新 `.nc` 是否包含 `rao`。
5. 检查 `excitation_force = Froude_Krylov_force + diffraction_force`。

当前验证结果显示：

| 算例 | 惯性矩阵最大相对误差 | 静水力矩阵最大相对误差 |
| --- | ---: | ---: |
| DM10 slender 单模块 | `4.38e-4` | `5.84e-7` |
| 10x10 方形模块首块 | `6.99e-3` | `1.07e-2` |

说明当前界面生成的 `.nc` 在格式、DOF 约定、RAO、静水力和惯性项上已经能和已有基准对齐。完整 10x10 动态辐射/绕射项需要同规模 100 浮体 BEM 计算才能严格数值对比。

## 13. 常见问题

### 13.1 点击计算后很久没有完成

检查问题数。100 浮体、多频率、细网格会非常慢。建议先用 1 个频率和较粗网格试算。

### 13.2 并行核数大于 1 时报 `joblib` 缺失

Capytaine 的并行 `solve_all(..., n_jobs>1)` 需要可选依赖 `joblib`。当前程序会在缺少 `joblib` 时自动降级为 `n_jobs=1`，不再中断计算。若希望启用并行，可在 Conda 环境中安装：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python -m pip install joblib
```

### 13.3 输出 `.nc` 找不到

检查右侧“输出文件”路径。如果是相对路径，它在项目根目录下，而不是浏览器下载目录。

### 13.4 想保留多个算例结果

每次计算前修改“NetCDF 文件”路径，避免覆盖上一次结果。

### 13.5 RAO 动画看起来太夸张

调小“运动倍率”。它只影响界面显示，不改变 `.nc` 中的水动力数据。

### 13.6 历史文件和新计算的 `rho` 不一致

旧 Yoon 数据中存在历史 notebook 约定：部分 `.nc` 的 `rho` 坐标、惯性质量、静水力计算密度并不总是完全一致。做验证时应优先复现历史质量和静水力约定；做新的物理计算时按实际海水密度填写。

### 13.7 如何把 `.nc` 传给 RODM

在 RODM 算例构造中把 `hydrodynamic_path` 指向新生成的 `.nc`。读取仍使用：

```python
from offshore_energy_sim.hydrodynamics import open_hydrodynamic_dataset

dataset = open_hydrodynamic_dataset("results/hydrodynamics_ui/your_case.nc")
```

后续 `prepare_hydrodynamic_terms()` 会读取附加质量、辐射阻尼、静水刚度和波浪力，并按 RODM 当前规则删除 yaw/rz。
