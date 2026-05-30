# Building a Windows executable (EXE) for eeg_app

This guide shows how to create a standalone Windows executable using PyInstaller.

Prerequisites:
- Python 3.10+ (use the same Python version as your virtualenv where the app runs)
- On Windows, install Microsoft Visual C++ Redistributable (2015-2022) if not already present.

Quick build (PowerShell) — runs the provided helper script:

```powershell
# From repository root
.\build_exe.ps1
```

The script will:
- create a temporary virtual environment `.venv_build`
- install `pyinstaller`
- run PyInstaller to produce either a one-file executable (default) or a folder build
- include `config/`, `models/` and `data/` directories into the bundle

Notes and troubleshooting:
- If your build fails due to missing DLLs (e.g. pyarrow), install those packages first inside the build venv by editing `build_exe.ps1` and uncommenting the `pip install` line.
- For large binary dependencies (pyarrow, numpy MKL), prefer a folder build (no `-F`) and ship the whole `dist\eeg_app` folder.
- Test the generated exe on a clean Windows machine and if it fails, check the missing DLLs with `Dependency Walker` or run the exe from cmd to view stdout.

Copying to another device:
- If you built a one-file exe, copy `dist\eeg_app.exe` to the other Windows machine and run it.
- If you built a folder, copy the entire `dist\eeg_app` directory.
- Ensure target machine has required VC++ runtime installed.
