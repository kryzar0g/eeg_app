<#
run_with_venv.ps1
Activate the project's `.venv310` and run the application.

Usage:
  .\run_with_venv.ps1         # activates .venv310 and runs run_app.py
  .\run_with_venv.ps1 -VenvPath ".venv310"
#>

param(
    [string]$VenvPath = ".venv310",
    [switch]$Gui = $true
)

$full = Join-Path $PWD $VenvPath
if (-not (Test-Path $full)) {
    Write-Error "Virtual environment not found: $full"
    exit 1
}

Write-Host "Activating venv: $full"
. "$full\Scripts\Activate.ps1"

if ($Gui) {
    python .\run_app.py
} else {
    py -3 .\run_app.py
}
