# Mac 本地 VS Code 开发环境说明

本文档记录本机 RODM 项目的本地开发配置。当前项目目录为：

```text
/Users/yongkang/Projects/RODM_20250310_local
```

## 1. Conda 环境

推荐使用已有环境：

```bash
conda activate offshore-energy-sim
```

如果当前终端找不到 `conda`，可直接使用完整路径：

```bash
/Users/yongkang/miniconda3/bin/conda activate offshore-energy-sim
```

VS Code 项目配置已经指向：

```text
/Users/yongkang/miniconda3/envs/offshore-energy-sim/bin/python
```

## 2. 已安装的编译与构建工具

这些工具已安装在 Conda 环境 `offshore-energy-sim` 中：

| 工具 | 用途 |
| --- | --- |
| `clang` | C 编译器 |
| `clang++` | C++ 编译器 |
| `gfortran` | Fortran 编译器 |
| `cmake` | 跨平台构建系统 |
| `ninja` | 快速构建后端 |
| `make` | 传统构建工具 |
| `pkg-config` | 查询库编译/链接参数 |

检查命令：

```bash
/Users/yongkang/miniconda3/bin/conda run -n offshore-energy-sim clang --version
/Users/yongkang/miniconda3/bin/conda run -n offshore-energy-sim gfortran --version
/Users/yongkang/miniconda3/bin/conda run -n offshore-energy-sim cmake --version
```

## 3. VS Code 使用方式

打开 VS Code 后，选择：

```text
File -> Open Folder -> /Users/yongkang/Projects/RODM_20250310_local
```

推荐安装 VS Code 扩展：

| 扩展 | 用途 |
| --- | --- |
| Python | Python 运行与调试 |
| Pylance | Python 代码智能提示 |
| Jupyter | Notebook 支持 |
| C/C++ | C/C++ IntelliSense |
| CMake Tools | CMake 项目支持 |
| Modern Fortran | Fortran 语法与 lint |

本项目已经提供 `.vscode/extensions.json`，VS Code 打开项目时会提示安装推荐扩展。

## 4. 常用任务

在 VS Code 中按 `Cmd+Shift+P`，输入 `Tasks: Run Task`，可直接运行：

| 任务 | 作用 |
| --- | --- |
| `Check: compiler versions` | 检查编译器和构建工具版本 |
| `Run: continuous wave validation` | 运行连续性浮体 60-300 m 波长验证流程 |
| `Run: Yoon hinge validation` | 运行单铰接/双铰接验证流程 |
| `Build: hydroelastic validation report` | 重新生成总验证报告 |

## 5. 调试入口

VS Code 的 Run and Debug 面板中已配置两个入口：

| 配置 | 作用 |
| --- | --- |
| `Python: current file` | 调试当前打开的 Python 文件 |
| `Python: Yoon hinge validation` | 直接调试 Yoon 铰接验证脚本 |

调试时会自动设置：

```text
PYTHONPATH=/Users/yongkang/Projects/RODM_20250310_local/src
```
