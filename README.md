# RODM 本地重构工作区

本目录是从 OneDrive 迁出的本机工作副本，用于把原始 RODM 研究代码逐步整理为一体化水弹性分析软件框架。

```text
/Users/yongkang/Projects/RODM_20250310_local
```

## 推荐阅读顺序

| 文件 | 用途 |
| --- | --- |
| `docs/user_guide_cn.md` | 使用者快速上手，包含常用运行命令。 |
| `docs/hydrodynamics_ui_user_guide_cn.md` | 水动力计算窗口说明：打开、参数设置、`.nc` 命名和验证。 |
| `docs/full_code_user_manual_cn.md` | 完整用户说明，解释每个文件夹和每个 `.py` 文件。 |
| `docs/code_structure_cn.md` | 代码包结构、各包职责和主流程图。 |
| `docs/code_structure_and_cleanup_cn.md` | 当前主线结构、清理记录和后续删除候选。 |
| `docs/code_line_notes_cn.md` | 面向开发者的关键函数说明。 |
| `docs/hydroelastic_validation_report.md` | 连续体浮体和铰接浮体验证总报告。 |
| `docs/local_vscode_setup_cn.md` | Mac + VS Code + Conda 本地开发说明。 |

## 当前主线代码

| 路径 | 说明 |
| --- | --- |
| `src/offshore_energy_sim/` | 标准化 Python 包，是后续一体化软件的核心代码。 |
| `scripts/` | 可重复运行的验证、批处理、报告生成入口。 |
| `configs/` | 配置驱动算例模板。 |
| `docs/` | 中文结构说明、验证报告、使用说明。 |
| `results/` | 本地运行输出、图片、响应数组和报告。 |
| `references/` | 论文图件、历史程序、数字化曲线等溯源资料。 |

根目录下的 `DM_*.py`、`SEREP.py`、`RODM_*.ipynb` 是历史研究脚本和论文复现记录。它们不作为新开发入口，但暂时保留用于数值溯源。

## Git 同步建议

当前工作区可以用 Git 同步到你的远程仓库。第一次同步通常按下面做：

```bash
cd /Users/yongkang/Projects/RODM_20250310_local
git init
git branch -M main
git add README.md .gitignore src scripts docs tests requirements.txt environment-mac.yml
git commit -m "Initialize RODM standardized hydrodynamics workflow"
git remote add origin <你的远程仓库地址>
git push -u origin main
```

如果远程仓库已经存在，先把仓库地址发给我，或者在本机提前配置好 SSH/token 权限，我就可以帮你执行 `git remote add`、`git pull`、`git push` 等同步操作。

不建议把本地大数据和计算结果直接放进 Git。当前 `.gitignore` 已默认忽略 `results/`、`.nc`、视频、VTK、Abaqus 大输出和本地数据镜像目录。外部数据建议继续放在独立数据目录，例如：

```text
/Users/yongkang/data/DM-FEM2D
D:\RODM-data\DM-FEM2D
```

## 常用命令

### macOS Conda 环境

建议统一使用本机 Conda 环境：

```bash
/Users/yongkang/miniconda3/bin/conda activate offshore-energy-sim
```

如果终端无法激活环境，可直接使用完整 Python 路径：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/check_environment.py
```

连续性浮体验证：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_regular_wave_batch_validation.py
```

单铰接/双铰接验证：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_yoon_hinge_cases.py --case all
```

10x10 模块铰接设置检查：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_complex_hinge_10x10.py --skip-solve
```

生成总验证报告：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/build_hydroelastic_validation_report.py
```

打开水动力计算窗口：

```bash
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python scripts/run_hydrodynamics_ui.py --host 127.0.0.1 --port 8765
```

### Windows 原生 Conda 环境

原始 RODM 代码可以继续在 Windows + Conda 中运行，不需要额外使用 Python `venv`，也不强制要求 WSL。建议在 Anaconda Prompt 或 PowerShell 中操作。

如果已有可用 Conda 环境，直接激活原环境并补齐依赖：

```bat
conda activate offshore-energy-sim
conda install -c conda-forge python=3.10 capytaine xarray netcdf4 h5netcdf h5py scipy pandas matplotlib pyyaml joblib
```

如果要新建一个干净 Conda 环境：

```bat
conda create -n offshore-energy-sim -c conda-forge python=3.10 capytaine xarray netcdf4 h5netcdf h5py scipy pandas matplotlib pyyaml joblib
conda activate offshore-energy-sim
```

在 Windows 项目目录运行：

```bat
cd D:\Code\RODM_20250310_local
python scripts\check_environment.py
python scripts\run_hydrodynamics_ui.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://localhost:8765/
```

如果外部数据放在 Windows 盘符中，可以在 PowerShell 设置：

```powershell
$env:RODM_DM_FEM_ROOT="D:\RODM-data\DM-FEM2D"
```

或在 CMD 设置：

```bat
set RODM_DM_FEM_ROOT=D:\RODM-data\DM-FEM2D
```

## 当前开发原则

- 新功能优先进入 `src/offshore_energy_sim/`，不要继续扩展旧 notebook。
- 新验证入口优先放入 `scripts/`，并输出到 `results/`。
- 不直接写入 OneDrive 原始目录。
- 不随意删除历史脚本、notebook、`.npy`、`.inp`、`.zip` 等溯源文件。
- 涉及 SEREP、DOF 顺序、铰接刚度、水动力节点顺序的改动必须先做验证。
