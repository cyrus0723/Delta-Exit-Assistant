$ErrorActionPreference = 'Stop'
$answer = Read-Host '是否创建桌面快捷方式？输入 Y 确认，其他输入取消'
if ($answer -notmatch '^[yY]$') { exit }
$target = Join-Path $PSScriptRoot 'Delta-Exit-Assistant.exe'
if (-not (Test-Path -LiteralPath $target)) { throw '请先完整解压压缩包。' }
$shell = New-Object -ComObject WScript.Shell
$desktop = $shell.SpecialFolders.Item('Desktop')
$link = $shell.CreateShortcut((Join-Path $desktop '三角洲下机助手.lnk'))
$link.TargetPath = $target
$link.WorkingDirectory = $PSScriptRoot
$link.IconLocation = "$target,0"
$link.Save()
Write-Host '桌面快捷方式已创建。'
