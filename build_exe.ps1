<#
.\build_exe.ps1
PowerShell helper to build a Windows executable using PyInstaller.

Usage (PowerShell):
  .\build_exe.ps1         # creates venv_build, installs PyInstaller, builds one-file exe
  .\build_exe.ps1 -KeepVenv # keep the created venv

Notes:
- The script bundles `run_app.py` as the entrypoint. It attempts to include `config/`, `models/` and `data/` directories.
- If your app depends on large binary wheels (e.g. pyarrow), ensure those are installed in the build venv first.
#>

param(
    [switch]$KeepVenv = $false,
    [string]$VenvName = ".venv_build",
    [switch]$NoOneFile = $false,
    [string]$ExistingVenv = ""  # if provided, use this existing venv path instead of creating a new one
)

$cwd = Split-Path -Path $PSScriptRoot -Parent
Write-Host "Working dir: $PWD"


if ($ExistingVenv -and (Test-Path $ExistingVenv)) {
    Write-Host "Using existing venv: $ExistingVenv"
    $venvPath = $ExistingVenv
} else {
    if (Test-Path $VenvName) {
        Write-Host "Removing existing venv: $VenvName"
        Remove-Item -Recurse -Force $VenvName
    }
    python -m venv $VenvName
    $venvPath = $VenvName
    # Install pyinstaller into the new venv
    . "$venvPath\Scripts\Activate.ps1"
    python -m pip install --upgrade pip
    python -m pip install pyinstaller
}

# Activate the selected venv
. "$venvPath\Scripts\Activate.ps1"

# Install runtime dependencies used by your app. Uncomment if you want to ensure they're present in build env.
# python -m pip install numpy scipy scikit-learn mne pyarrow joblib

Write-Host "Running PyInstaller..."

$addData = @( 
    "config;config",
    "models;models",
    "data;data"
)

$addDataArgs = $addData | ForEach-Object { "--add-data `"$($_)`"" }
$addDataArgsStr = $addDataArgs -join ' '

# Build argument array for PyInstaller to avoid quoting issues
$pyArgs = @('-y')
if (-not $NoOneFile) {
    $pyArgs += '-F'
}
$pyArgs += '--name'
$pyArgs += 'eeg_app'

foreach ($d in $addData) {
    $pyArgs += '--add-data'
    $pyArgs += $d
}

# hidden import for problematic optional dependency
$pyArgs += '--hidden-import'
$pyArgs += 'pyarrow'

# Entry script is run_app.py at repo root
$pyArgs += 'run_app.py'

Write-Host "pyinstaller arguments: $($pyArgs -join ' ')"
& pyinstaller @pyArgs

Write-Host "Build finished. See the 'dist' directory for the built artifact."

if (-not $KeepVenv) {
    Write-Host "Removing build venv..."
    try { Deactivate } catch {}
    Remove-Item -Recurse -Force $VenvName
}

Write-Host "Done."
