# GitHub 协同工作流说明

本文件用于说明 MacBook / 笔记本与 Windows 工作站之间的 GitHub 协同方式。核心原则是：代码进入 GitHub，大型科研数据留在工作站数据盘。

## 基本原则

1. 代码、配置、脚本、测试、文档草稿和最终论文图脚本可以进入 GitHub。
2. 大型数据不要进入 GitHub，包括 hydrodynamic 数据、`.npy`、`.npz`、`.mat`、`.h5`、`.hdf5`、`.vtk`、`.vtu` 等。
3. MacBook 上可以使用 Codex 修改 GitHub 仓库，完成后提交和推送。
4. Windows 工作站通过 `git pull` 获取最新代码，然后在本地 Conda 环境中运行验证。
5. 验证结果只保存关键图表、小型表格和必要摘要，避免把完整计算输出全部提交。
6. 大型 hydrodynamic / npy / mat / h5 / vtk 数据建议保留在 `D:\ResearchData`。
7. 文档、小型结果和最终图可按需要通过 OneDrive / Syncthing 同步。

## 推荐工作流

### MacBook / 笔记本端

1. 从 GitHub 拉取仓库：

```bash
git clone <repo-url>
```

2. 使用 Codex 或 VS Code 修改代码、配置和文档。
3. 本地运行轻量检查或单元测试。
4. 提交并推送：

```bash
git status
git add <changed-files>
git commit -m "Describe the change"
git push
```

### Windows 工作站端

1. 进入工作站代码目录，例如：

```powershell
cd D:\ResearchCode\OFPV_RODM
```

2. 拉取最新代码：

```powershell
git pull
```

3. 激活 Conda 环境并运行验证：

```powershell
conda activate ofpv
python main.py
```

4. 长时间任务建议使用 `run_long_python_task_template.ps1`，把输出写入 `logs`。
5. 验证通过后，只整理必要的小型结果、最终图表或摘要文件，再决定是否提交。

## 推荐 .gitignore 示例

以下内容适合放入项目根目录 `.gitignore`，用于避免误提交缓存、虚拟环境、大型数据和临时输出：

```gitignore
__pycache__/
*.pyc
.venv/
.env
*.npy
*.npz
*.mat
*.h5
*.hdf5
*.vtk
*.vtu
results/
outputs/
logs/
.idea/
.vscode/
```

注意：不要一刀切忽略所有科研成果。如果某些小型结果图、论文最终图、表格摘要需要版本管理，可以单独放入 `figures/final/`，并通过 `.gitignore` 例外规则纳入 Git。

例如：

```gitignore
figures/*
!figures/final/
!figures/final/**
```

## 数据目录建议

大型数据建议放在：

```text
D:\ResearchData\Hydrodynamic
D:\ResearchData\RODM_Results
D:\ResearchData\OFPV_Power_Results
D:\ResearchData\TimeDomain_Results
```

这些目录不建议作为 Git 仓库的一部分。若需要记录数据来源和生成方式，可以在 GitHub 中保存轻量 README、配置文件、生成脚本和 checksum 摘要。

## 协同注意事项

- 在 MacBook 端修改代码前先 `git pull`；
- 在 Windows 工作站运行验证前也先 `git pull`；
- 不要在两个设备上同时修改同一文件后长期不提交；
- 不要提交密码、token、私钥或个人配置；
- 大型结果只在需要时整理成小型最终图表或摘要文件；
- 实验复现依赖代码、配置和日志，而不是把所有中间文件都提交到 GitHub。
