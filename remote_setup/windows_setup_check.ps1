# Windows workstation remote setup checker.
# This script only checks status. It does not modify system settings.

$ErrorActionPreference = 'Continue'

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

Write-Host ''
Write-Host '=== Windows 工作站远程协同环境检查 ===' -ForegroundColor Cyan
Write-Host ''

Write-Status 'INFO' "当前 Windows 用户名：$env:USERNAME"
Write-Status 'INFO' "当前计算机名：$env:COMPUTERNAME"
Write-Status 'INFO' "PowerShell 版本：$($PSVersionTable.PSVersion)"

Write-Host ''
Write-Host '--- OpenSSH Server ---' -ForegroundColor Cyan

$openSshInstalled = $false
$sshdExePath = Join-Path -Path $env:WINDIR -ChildPath 'System32\OpenSSH\sshd.exe'
try {
    $openSshCapability = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' -ErrorAction Stop
    if ($openSshCapability.State -eq 'Installed') {
        $openSshInstalled = $true
        Write-Status 'OK' 'OpenSSH Server 已安装'
    }
    elseif (Test-Path -LiteralPath $sshdExePath) {
        $openSshInstalled = $true
        Write-Status 'OK' "检测到 sshd.exe，OpenSSH Server 可能已安装：$sshdExePath"
    }
    else {
        Write-Status 'WARN' "OpenSSH Server 未安装或状态异常：$($openSshCapability.State)"
    }
}
catch {
    if (Test-Path -LiteralPath $sshdExePath) {
        $openSshInstalled = $true
        Write-Status 'OK' "检测到 sshd.exe，OpenSSH Server 可能已安装：$sshdExePath"
    }
    else {
        Write-Status 'WARN' "无法通过 Get-WindowsCapability 检查 OpenSSH Server，且未检测到 sshd.exe：$($_.Exception.Message)"
    }
}

$sshdService = Get-Service -Name 'sshd' -ErrorAction SilentlyContinue
if ($null -eq $sshdService) {
    Write-Status 'WARN' 'sshd 服务不存在'
}
else {
    Write-Status 'OK' 'sshd 服务已存在'

    if ($sshdService.Status -eq 'Running') {
        Write-Status 'OK' 'sshd 服务正在运行'
    }
    else {
        Write-Status 'WARN' "sshd 服务未运行，当前状态：$($sshdService.Status)"
    }

    try {
        $sshdConfig = Get-CimInstance -ClassName Win32_Service -Filter "Name='sshd'" -ErrorAction Stop
        if ($sshdConfig.StartMode -eq 'Auto') {
            Write-Status 'OK' 'sshd 服务已设置为开机自启动'
        }
        else {
            Write-Status 'WARN' "sshd 服务未设置为开机自启动，当前启动类型：$($sshdConfig.StartMode)"
        }
    }
    catch {
        Write-Status 'WARN' "无法检查 sshd 服务启动类型：$($_.Exception.Message)"
    }
}

if (-not $openSshInstalled -and $null -ne $sshdService) {
    Write-Status 'INFO' '虽然未通过 Windows Capability 检测到 OpenSSH Server，但 sshd 服务存在，请结合实际安装方式判断。'
}

Write-Host ''
Write-Host '--- Windows 防火墙 ---' -ForegroundColor Cyan

try {
    $sshNamedRules = Get-NetFirewallRule -Direction Inbound -ErrorAction Stop |
        Where-Object {
            $_.DisplayName -match 'OpenSSH|sshd|SSH' -or
            $_.Name -match 'OpenSSH|sshd|SSH'
        }

    if ($sshNamedRules) {
        Write-Status 'OK' 'Windows 防火墙中检测到 OpenSSH / sshd / SSH 相关入站规则'
        $sshNamedRules |
            Select-Object Name, DisplayName, Enabled, Action, Profile |
            Format-Table -AutoSize
    }
    else {
        Write-Status 'WARN' 'Windows 防火墙中未检测到 OpenSSH / sshd / SSH 相关入站规则'
    }

    $tcp22AllowRules = Get-NetFirewallRule -Direction Inbound -Enabled True -Action Allow -ErrorAction Stop |
        Get-NetFirewallPortFilter |
        Where-Object {
            $ports = @($_.LocalPort)
            $_.Protocol -eq 'TCP' -and ($ports -contains '22' -or $ports -contains 'Any')
        }

    if ($tcp22AllowRules) {
        Write-Status 'OK' 'Windows 防火墙已存在允许 TCP 22 入站的启用规则'
    }
    else {
        Write-Status 'WARN' '未检测到已启用的 TCP 22 入站允许规则'
    }
}
catch {
    Write-Status 'WARN' "无法通过 Get-NetFirewallRule 检查 Windows 防火墙规则：$($_.Exception.Message)"
    Write-Status 'INFO' '普通 PowerShell 可能无法完整读取防火墙规则；如需完整结果，可用管理员 PowerShell 运行本检查脚本。'

    try {
        $netshOutput = (& netsh advfirewall firewall show rule name=all) 2>$null
        $netshMatches = $netshOutput | Select-String -Pattern 'OpenSSH|sshd|SSH'

        if ($netshMatches) {
            Write-Status 'OK' 'netsh 检测到 SSH 相关防火墙规则片段：'
            $netshMatches |
                Select-Object -First 12 |
                ForEach-Object { Write-Host "  $($_.Line.Trim())" }
        }
        else {
            Write-Status 'WARN' 'netsh 未检测到 OpenSSH / sshd / SSH 相关防火墙规则'
        }
    }
    catch {
        Write-Status 'WARN' "netsh 防火墙规则检查也失败：$($_.Exception.Message)"
    }
}

Write-Host ''
Write-Host '--- 开发工具 ---' -ForegroundColor Cyan

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($gitCommand) {
    $gitVersion = (& git --version) 2>$null
    Write-Status 'OK' "Git 可用：$gitVersion"
}
else {
    Write-Status 'ERROR' '未检测到 Git，请先安装 Git for Windows'
}

$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCommand) {
    $condaVersion = (& conda --version) 2>$null
    Write-Status 'OK' "Conda 可用：$condaVersion"
}
else {
    Write-Status 'WARN' '未检测到 Conda。若需要 Conda 环境，请安装 Miniconda / Anaconda，或确认 conda 已加入当前 PowerShell 环境'
}

Write-Host ''
Write-Host '--- IP 地址 ---' -ForegroundColor Cyan

try {
    $ipv4Addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object {
            $_.IPAddress -notmatch '^127\.' -and
            $_.IPAddress -notmatch '^169\.254\.'
        } |
        Sort-Object InterfaceAlias, IPAddress

    if ($ipv4Addresses) {
        Write-Status 'OK' '当前检测到的 IPv4 地址如下：'
        $ipv4Addresses |
            Select-Object InterfaceAlias, IPAddress, PrefixLength |
            Format-Table -AutoSize

        $tailscaleCandidates = $ipv4Addresses | Where-Object { $_.IPAddress -like '100.*' }
        if ($tailscaleCandidates) {
            Write-Status 'OK' '检测到疑似 Tailscale 100.x.x.x 地址，请优先使用该地址进行远程 SSH 连接'
        }
        else {
            Write-Status 'INFO' '未在系统 IP 列表中看到 100.x.x.x 地址，请打开 Tailscale 客户端查看本机 Tailscale IP'
        }
    }
    else {
        Write-Status 'WARN' '未检测到可用 IPv4 地址'
    }
}
catch {
    Write-Status 'WARN' "无法检查当前 IP 地址：$($_.Exception.Message)"
}

Write-Host ''
Write-Status 'INFO' '请在 Tailscale 软件中查看 Windows 工作站的 100.x.x.x Tailscale IP，并在 MacBook 端用于 SSH / VS Code Remote SSH。'
Write-Host ''


