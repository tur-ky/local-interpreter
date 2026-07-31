# Build Local Interpreter: icon -> frozen app -> installer.
#
#   .\build.ps1              full build
#   .\build.ps1 -SkipApp     only recompile the installer from dist\
#
param(
    [switch]$SkipApp,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-ISCC {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    throw "Inno Setup 6 not found. Install it with: winget install JRSoftware.InnoSetup"
}

if (-not $SkipApp) {
    Write-Host "==> icon" -ForegroundColor Cyan
    python make_icon.py

    Write-Host "==> PyInstaller" -ForegroundColor Cyan
    python -m PyInstaller --noconfirm --clean LocalInterpreter.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

    Write-Host "==> self-test of the frozen build" -ForegroundColor Cyan
    & ".\dist\LocalInterpreter\LocalInterpreterFetch.exe" --self-test
    if ($LASTEXITCODE -ne 0) { Write-Warning "self-test reported problems (models may simply be missing)" }
}

if ($SkipInstaller) {
    Write-Host "==> installer skipped" -ForegroundColor Cyan
    return
}

Write-Host "==> Inno Setup" -ForegroundColor Cyan
& (Find-ISCC) installer.iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

Get-ChildItem installer\*.exe | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
