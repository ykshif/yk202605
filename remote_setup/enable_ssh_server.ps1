# Enable Windows OpenSSH Server for remote development.
# Run this script in an elevated PowerShell window.

$ErrorActionPreference = 'Stop'

function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('OK', 'WARN', 'ERROR', 'INFO')]
        [string]$Level,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $color = switch ($Level) {
        'OK' { 'Green' }
        'WARN' { 'Yellow' }
        'ERROR' { 'Red' }
        default { 'Cyan' }
    }

    Write-Host "[$Level] $Message" -ForegroundColor $color
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host ''
Write-Host '=== 启用 Windows OpenSSH Server ===' -ForegroundColor Cyan
Write-Host ''

if (-not (Test-IsAdministrator)) {
    Write-Status 'ERROR' '当前 PowerShell 不是管理员权限。请右键 PowerShell，选择“以管理员身份运行”，然后重新执行本脚本。'
    exit 1
}

Write-Status 'OK' '已确认当前 PowerShell 具有管理员权限'

$openSshInstalled = $false
$sshdService = Get-Service -Name 'sshd' -ErrorAction SilentlyContinue

try {
    $openSshCapability = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' -ErrorAction Stop
    if ($openSshCapability.State -eq 'Installed') {
        $openSshInstalled = $true
        Write-Status 'OK' 'OpenSSH Server 已安装'
    }
    else {
        Write-Status 'WARN' "OpenSSH Server 当前状态：$($openSshCapability.State)"
    }
}
catch {
    Write-Status 'WARN' "无法通过 Get-WindowsCapability 检查 OpenSSH Server：$($_.Exception.Message)"
}

if (-not $openSshInstalled -and $null -eq $sshdService) {
    Write-Status 'ERROR' '未检测到 OpenSSH Server 或 sshd 服务。请先手动安装 OpenSSH Server。'
    Write-Host ''
    Write-Host '安装路径：' -ForegroundColor Cyan
    Write-Host '设置 → 系统 → 可选功能 → 查看功能 → 搜索 OpenSSH Server → 安装'
    Write-Host ''
    Write-Host '安装完成后，请重新以管理员 PowerShell 运行：'
    Write-Host '.\remote_setup\enable_ssh_server.ps1'
    exit 1
}

try {
    Start-Service -Name 'sshd'
    Write-Status 'OK' 'sshd 服务已启动'

    Set-Service -Name 'sshd' -StartupType Automatic
    Write-Status 'OK' 'sshd 服务已设置为开机自启动'
}
catch {
    Write-Status 'ERROR' "启动或配置 sshd 服务失败：$($_.Exception.Message)"
    exit 1
}

try {
    $tcp22AllowRules = Get-NetFirewallRule -Direction Inbound -Enabled True -Action Allow -ErrorAction Stop |
        Get-NetFirewallPortFilter |
        Where-Object {
            $ports = @($_.LocalPort)
            $_.Protocol -eq 'TCP' -and ($ports -contains '22' -or $ports -contains 'Any')
        }

    if ($tcp22AllowRules) {
        Write-Status 'OK' '已存在允许 TCP 22 入站的防火墙规则'
    }
    else {
        $ruleName = 'OpenSSH-Server-In-TCP'
        $existingOpenSshRule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue

        if ($existingOpenSshRule) {
            Set-NetFirewallRule -Name $ruleName -Enabled True -Action Allow
            Write-Status 'OK' "已启用已有防火墙规则：$ruleName"
        }
        else {
            New-NetFirewallRule `
                -Name $ruleName `
                -DisplayName 'OpenSSH Server (sshd)' `
                -Enabled True `
                -Direction Inbound `
                -Protocol TCP `
                -Action Allow `
                -LocalPort 22 | Out-Null

            Write-Status 'OK' '已创建防火墙入站规则，允许 TCP 22 端口'
        }
    }
}
catch {
    Write-Status 'ERROR' "检查或创建防火墙规则失败：$($_.Exception.Message)"
    exit 1
}

Write-Host ''
Write-Status 'OK' 'OpenSSH Server 启用流程完成'
Write-Host ''
Write-Host 'MacBook / 笔记本连接示例：' -ForegroundColor Cyan
Write-Host "ssh $env:USERNAME@100.xxx.xxx.xxx"
Write-Host ''
Write-Host '请将 100.xxx.xxx.xxx 替换为 Windows 工作站在 Tailscale 中显示的 100.x.x.x IP。'
Write-Host '本脚本不会要求输入密码，也不会保存任何密码、token 或密钥。'
Write-Host ''


