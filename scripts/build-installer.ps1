[CmdletBinding()]
param(
    [string]$Version = "2.3.0"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE "AppData\Local" }
$python = Join-Path $localAppData "Programs\Python\Python312\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 Python 3.12。请安装 Python 3.12 与 requirements.txt 中的依赖后重试。"
}

$releaseRoot = Join-Path $projectRoot "release"
$appDist = Join-Path $releaseRoot "app"
$buildDir = Join-Path $releaseRoot "build"
$specDir = Join-Path $releaseRoot "spec"
$assetsDir = Join-Path $projectRoot "assets"
$iconPath = Join-Path $assetsDir "icon.ico"

Push-Location $projectRoot
try {
    & $python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name "Delta-Exit-Assistant" `
        --paths "src" `
        --add-data "$assetsDir;assets" `
        --icon $iconPath `
        --distpath $appDist `
        --workpath $buildDir `
        --specpath $specDir `
        "src\app.py"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 打包失败，退出代码：$LASTEXITCODE"
    }

    $isccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    $isccPath = if ($isccCommand) { $isccCommand.Source } else { $null }
    if (-not $isccPath) {
        $standardPaths = @(
            (Join-Path $localAppData "Programs\Inno Setup 6\ISCC.exe"),
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 6\ISCC.exe"
        )
        foreach ($candidate in $standardPaths) {
            if (Test-Path -LiteralPath $candidate) {
                $isccPath = $candidate
                break
            }
        }
    }
    if (-not $isccPath) {
        throw "未找到 Inno Setup 6。请安装后重新运行此脚本。"
    }

    $env:DELTA_EXIT_ASSISTANT_VERSION = $Version
    & $isccPath "installer\Delta-Exit-Assistant.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup 编译失败，退出代码：$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
