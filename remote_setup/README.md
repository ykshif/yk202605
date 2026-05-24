# Windows 工作站远程协同配置

本目录用于建立“MacBook / 笔记本随身控制端 + Windows 工作站算力端”的远程协同开发环境。这里的脚本和文档只服务于远程连接、环境检查、目录规划和长时间任务运行，不涉及 RODM、OFPV、TimeDomain 等科研主程序修改。

## 整体架构

```text
MacBook / 笔记本
    ↓
Tailscale 私有网络
    ↓
Windows 工作站
    ├── VS Code Remote SSH：远程开发和运行 Python
    ├── Windows Remote Desktop / RustDesk：远程桌面控制
    ├── GitHub：代码版本管理
    └── OneDrive / Syncthing：文档和小型结果同步
```

推荐把 Windows 工作站作为主要算力端，用于运行 Python、仿真、数据处理和长期任务；MacBook 或轻薄笔记本作为随身控制端，通过 Tailscale + SSH + VS Code 连接工作站。

## 推荐配置顺序

1. 在 Windows 工作站安装并登录 Tailscale，确认设备在线。
2. 用普通 PowerShell 运行 `windows_setup_check.ps1`，检查 SSH、Git、Conda、防火墙和 IP 状态。
3. 如需启用 SSH，用管理员 PowerShell 运行 `enable_ssh_server.ps1`。
4. 用普通 PowerShell 运行 `create_research_dirs.ps1`，在 D 盘创建推荐科研目录结构。
5. 在 MacBook 上配置 VS Code Remote SSH，连接 Windows 工作站的 Tailscale IP。
6. 使用 GitHub 同步代码，大型数据保留在 Windows 工作站本地数据盘。

## Windows 工作站端配置步骤

### 1. 检查当前状态

在项目根目录运行：

```powershell
.\remote_setup\windows_setup_check.ps1
```

该脚本只做检查，不会自动修改系统。重点查看：

- OpenSSH Server 是否安装；
- `sshd` 服务是否存在、运行、开机自启动；
- Windows 防火墙是否存在 SSH / TCP 22 入站规则；
- Git 和 Conda 是否可用；
- 当前局域网 IP，并提示你在 Tailscale 中查看 `100.x.x.x` 地址。

普通 PowerShell 下，少数 Windows 系统能力和防火墙详情可能因为权限不足只能给出部分结果；如果需要完整检查，可以用管理员 PowerShell 再运行一次该检查脚本。

### 2. 启用 OpenSSH Server

如果检查发现 SSH 未启用，请以管理员身份打开 PowerShell，并运行：

```powershell
.\remote_setup\enable_ssh_server.ps1
```

脚本会检查管理员权限、启动 `sshd` 服务、设置开机自启动，并创建或确认 TCP 22 防火墙入站规则。脚本不会要求输入密码，也不会保存任何密码、token 或密钥。

如果 OpenSSH Server 没有安装，请手动安装：

```text
设置 → 系统 → 可选功能 → 查看功能 → 搜索 OpenSSH Server → 安装
```

安装完成后再重新运行 `enable_ssh_server.ps1`。

### 3. 创建科研目录结构

在普通 PowerShell 中运行：

```powershell
.\remote_setup\create_research_dirs.ps1
```

该脚本会在 D 盘创建：

```text
D:\ResearchCode
D:\ResearchData
D:\ResearchDocs
D:\ResearchBackup
```

以及若干推荐子目录。已有目录会跳过，不会删除或覆盖任何文件。

### 4. 长时间 Python 任务

复制或修改 `run_long_python_task_template.ps1` 中的变量：

- `$CondaEnvName`：Conda 环境名；
- `$PythonScriptPath`：要运行的 Python 脚本路径；
- `$WorkingDirectory`：任务工作目录；
- `$ScriptArguments`：可选命令行参数。

脚本会自动创建 `logs` 目录，并把输出保存到带日期时间的日志文件。

## MacBook 端连接方式

### 1. 确认 Tailscale 在线

在 MacBook 和 Windows 工作站上都登录同一个 Tailscale 账号或同一个 tailnet，确认 Windows 工作站显示在线，并记录其 `100.x.x.x` Tailscale IP。

### 2. 直接 SSH 测试

在 MacBook 终端中测试：

```bash
ssh Windows用户名@100.xxx.xxx.xxx
```

例如：

```bash
ssh WYJ@100.xxx.xxx.xxx
```

第一次连接会提示确认主机指纹，并要求输入 Windows 登录密码。

### 3. VS Code Remote SSH

参考 `vscode_remote_ssh_config_example.txt`，在 MacBook 的 SSH 配置中添加：

```text
Host yk-windows-workstation
    HostName 100.xxx.xxx.xxx
    User WYJ
    Port 22
```

然后在 VS Code 中安装并使用 Remote - SSH 扩展，连接 `yk-windows-workstation`。连接成功后建议打开 `D:\ResearchCode` 下的代码目录。

### 4. 远程桌面控制

SSH 适合写代码、运行命令和管理长期任务。若需要完整桌面控制，可以搭配：

- Windows Remote Desktop；
- RustDesk；
- Tailscale 的私有网络地址。

远程桌面只用于图形界面操作，科研代码开发和运行建议仍以 VS Code Remote SSH 为主。

## GitHub 协同建议

- 代码、配置、论文图脚本进入 GitHub；
- 大型仿真数据、`.npy`、`.mat`、`.h5`、`.vtk` 等不要进入 GitHub；
- MacBook 上可以使用 Codex 修改仓库并提交；
- Windows 工作站上 `git pull` 后运行验证；
- 关键图表和小型结果可以单独整理后版本管理；
- 大型数据保留在 `D:\ResearchData`。

更详细的工作流见 `git_workflow_notes.md`。

## 常见问题排查

### SSH 连接超时

优先检查：

1. MacBook 和 Windows 工作站是否都在 Tailscale 中在线；
2. `HostName` 是否填写 Windows 工作站的 `100.x.x.x` Tailscale IP；
3. Windows 上 `sshd` 服务是否正在运行；
4. Windows 防火墙是否允许 TCP 22 入站。

### 提示密码错误

确认使用的是 Windows 本机用户名和 Windows 登录密码。用户名可在 Windows 上运行 `whoami` 或 `.\remote_setup\windows_setup_check.ps1` 查看。

### VS Code Remote SSH 连接失败

先用 MacBook 终端直接运行 `ssh 用户名@Tailscale-IP`。如果终端也失败，说明问题在网络、账号、服务或防火墙；如果终端成功但 VS Code 失败，再检查 VS Code Remote SSH 配置。

### `conda activate` 不生效

在 PowerShell 中先运行：

```powershell
conda init powershell
```

然后关闭当前 PowerShell 窗口，重新打开后再运行任务脚本。

### Git 不可用

安装 Git for Windows 后重新打开 PowerShell，再运行检查脚本。

## 安全注意事项

- 不要把 Windows 工作站的 SSH 端口直接暴露到公网，优先通过 Tailscale 私有网络访问；
- 使用强 Windows 登录密码；
- 不要在脚本、文档或仓库中保存密码、token、私钥；
- 只把必要设备加入 Tailscale，并定期检查设备列表；
- 防火墙规则只开放必要端口；
- 不要把大型数据、隐私数据或临时结果误提交到 GitHub；
- 长时间任务建议写日志，便于远程排查和复现实验。

## 边界说明

`remote_setup/` 只用于远程协同配置、环境检查、目录建议和运行模板，不修改本项目已有科研代码，不调整 RODM、OFPV、TimeDomain 等核心程序。
