# Thin PowerShell wrapper for Stable Fast 3D setup on Windows.
# Prefer Setup & Status in the viewer; this is for people who want a double-clickable
# companion to the Mac/Linux shell scripts.
#
#   .\scripts\bootstrap_sf3d.ps1
#   .\scripts\bootstrap_sf3d.ps1 -Yes
#
# Needs: NVIDIA GPU, Python env from install.ps1, and Visual Studio Build Tools (C++)
# to compile SF3D's extensions. Full detail: docs\WINDOWS.md

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

$argsList = @((Join-Path $Root "scripts\bootstrap_sf3d.py"))
if ($Yes) { $argsList += "--yes" }
if ($CodeOnly) { $argsList += "--code-only" }
if ($WeightsOnly) { $argsList += "--weights-only" }

& $Py @argsList
exit $LASTEXITCODE
