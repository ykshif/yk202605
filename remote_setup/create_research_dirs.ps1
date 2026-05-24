# Create recommended research directories on D:.
# This script creates missing directories only. It never deletes files.

$ErrorActionPreference = 'Stop'

function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('OK', 'WARN', 'ERROR', 'INFO', 'SKIP')]
        [string]$Level,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $color = switch ($Level) {
        'OK' { 'Green' }
        'WARN' { 'Yellow' }
        'ERROR' { 'Red' }
        'SKIP' { 'DarkGray' }
        default { 'Cyan' }
    }

    Write-Host "[$Level] $Message" -ForegroundColor $color
}

Write-Host ''
Write-Host '=== 创建推荐科研目录结构 ===' -ForegroundColor Cyan
Write-Host ''

if (-not (Test-Path -LiteralPath 'D:\')) {
    Write-Status 'ERROR' '未检测到 D 盘。请确认数据盘存在后再运行本脚本。'
    exit 1
}

$directories = @(
    'D:\ResearchCode',
    'D:\ResearchCode\OFPV_RODM',
    'D:\ResearchCode\OFPV_Power_Model',
    'D:\ResearchCode\TimeDomain_Adapter',
    'D:\ResearchCode\PanelMethod_AI',
    'D:\ResearchData',
    'D:\ResearchData\Hydrodynamic',
    'D:\ResearchData\RODM_Results',
    'D:\ResearchData\OFPV_Power_Results',
    'D:\ResearchData\TimeDomain_Results',
    'D:\ResearchDocs',
    'D:\ResearchDocs\Papers',
    'D:\ResearchDocs\Proposals',
    'D:\ResearchDocs\PPT',
    'D:\ResearchDocs\CV',
    'D:\ResearchBackup'
)

foreach ($directory in $directories) {
    $existingItem = Get-Item -LiteralPath $directory -ErrorAction SilentlyContinue

    if ($existingItem -and $existingItem.PSIsContainer) {
        Write-Status 'SKIP' "目录已存在，跳过：$directory"
        continue
    }

    if ($existingItem -and -not $existingItem.PSIsContainer) {
        Write-Status 'WARN' "路径已存在但不是目录，请手动检查：$directory"
        continue
    }

    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    Write-Status 'OK' "已创建目录：$directory"
}

Write-Host ''
Write-Status 'OK' '科研目录结构检查与创建完成。未删除、移动或覆盖任何已有文件。'
Write-Host ''


