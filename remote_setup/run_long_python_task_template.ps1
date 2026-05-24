# Long-running Python task template for Windows PowerShell.
#
# Before using conda activate in PowerShell, you may need to run once:
#   conda init powershell
# Then close PowerShell and open it again.

$ErrorActionPreference = 'Stop'

# 修改这里：Conda 环境名称。
$CondaEnvName = 'ofpv'

# 修改这里：要运行的 Python 脚本路径，可以是相对路径或绝对路径。
$PythonScriptPath = 'main.py'

# 修改这里：任务工作目录。默认使用当前目录。
$WorkingDirectory = (Get-Location).Path

# 修改这里：如果 Python 脚本需要参数，在这里填写，例如 @('--case', 'demo')。
$ScriptArguments = @()

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
Write-Host '=== 长时间 Python 任务运行模板 ===' -ForegroundColor Cyan
Write-Host ''

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Status 'ERROR' '未检测到 conda。请确认已安装 Miniconda / Anaconda，并已在当前 PowerShell 中初始化。'
    Write-Status 'INFO' '如果 conda activate 不生效，请先运行：conda init powershell，然后重新打开 PowerShell。'
    exit 1
}

Set-Location -LiteralPath $WorkingDirectory
Write-Status 'INFO' "当前任务工作目录：$WorkingDirectory"

if (-not (Test-Path -LiteralPath $PythonScriptPath)) {
    Write-Status 'ERROR' "Python 脚本不存在，请修改 `$PythonScriptPath：$PythonScriptPath"
    exit 1
}

$LogDirectory = Join-Path -Path $WorkingDirectory -ChildPath 'logs'
if (-not (Test-Path -LiteralPath $LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    Write-Status 'OK' "已创建日志目录：$LogDirectory"
}
else {
    Write-Status 'INFO' "日志目录已存在：$LogDirectory"
}

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path -Path $LogDirectory -ChildPath "run_$Timestamp.txt"

Write-Status 'INFO' "准备激活 Conda 环境：$CondaEnvName"
conda activate $CondaEnvName

if (-not $?) {
    Write-Status 'ERROR' "conda activate 失败：$CondaEnvName"
    Write-Status 'INFO' '如果 conda activate 不生效，请先运行：conda init powershell，然后重新打开 PowerShell。'
    exit 1
}

Write-Status 'OK' "Conda 环境已激活：$CondaEnvName"
Write-Status 'INFO' "开始运行 Python 脚本，日志文件：$LogFile"

& python $PythonScriptPath @ScriptArguments > $LogFile 2>&1
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Status 'OK' "Python 任务完成，日志已保存：$LogFile"
}
else {
    Write-Status 'ERROR' "Python 任务退出码：$exitCode。请查看日志：$LogFile"
}

exit $exitCode


