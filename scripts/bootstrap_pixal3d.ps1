# Thin PowerShell wrapper for Pixal3D setup on Windows.
# Prefer Setup & Status in the viewer; this mirrors the one-line shell helpers on Mac/Linux.
#
#   .\scripts\bootstrap_pixal3d.ps1
#   .\scripts\bootstrap_pixal3d.ps1 -Yes
#
# Downloads the CUDA 12 prebuilt when the driver is 575+. Full detail: docs\WINDOWS.md

param(
    [switch]$Yes,
    [switch]$BuildOnly,
    [switch]$WeightsOnly,
    [switch]$Prebuilt
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "No .venv yet. Run install.ps1 first (see docs\WINDOWS.md)." -ForegroundColor Red
    exit 1
}

$argsList = @((Join-Path $Root "scripts\bootstrap_pixal3d.py"))
if ($Yes) { $argsList += "--yes" }
if ($BuildOnly) { $argsList += "--build-only" }
if ($WeightsOnly) { $argsList += "--weights-only" }
if ($Prebuilt) { $argsList += "--prebuilt" }

& $Py @argsList
exit $LASTEXITCODE
