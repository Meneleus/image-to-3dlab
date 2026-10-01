# Thin PowerShell wrapper for official TRELLIS.2 CUDA setup on Windows.
# Prefer Setup & Status in the viewer. Full detail: docs\WINDOWS.md
# The Python script sets MSVC/CUDA C++20 flags — no hand-set CXXFLAGS needed.
#
#   .\scripts\bootstrap_trellis_cuda.ps1
#   .\scripts\bootstrap_trellis_cuda.ps1 -Yes

param([switch]$Yes)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "No .venv yet. Run install.ps1 first (see docs\WINDOWS.md)." -ForegroundColor Red
    exit 1
}
$argsList = @((Join-Path $Root "scripts\bootstrap_trellis_cuda.py"))
if ($Yes) { $argsList += "--yes" }
& $Py @argsList
exit $LASTEXITCODE
