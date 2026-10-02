# Thin PowerShell wrapper for official Hunyuan3D-2.1 CUDA setup on Windows.
# Prefer Setup & Status in the viewer. Full detail: docs\WINDOWS.md
#
#   .\scripts\bootstrap_hunyuan_cuda.ps1
#   .\scripts\bootstrap_hunyuan_cuda.ps1 -Yes
#   .\scripts\bootstrap_hunyuan_cuda.ps1 -Yes -CodeOnly

param(
    [switch]$Yes,
    [switch]$CodeOnly,
    [switch]$WeightsOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "No .venv yet. Run install.ps1 first (see docs\WINDOWS.md)." -ForegroundColor Red
    exit 1
}
$argsList = @((Join-Path $Root "scripts\bootstrap_hunyuan_cuda.py"))
if ($Yes) { $argsList += "--yes" }
if ($CodeOnly) { $argsList += "--code-only" }
if ($WeightsOnly) { $argsList += "--weights-only" }
& $Py @argsList
exit $LASTEXITCODE
