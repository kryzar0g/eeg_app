"""
Sestaví standalone EXE soubor pomocí PyInstaller.

Spuštění:
    python build_exe.py

Výstup:
    dist/eeg_app/eeg_app.exe   (složka s EXE a všemi závislostmi)
    dist/eeg_app_portable/     (to samé, přejmenováno pro distribuci)

Poznámky:
    • PsychoPy vyžaduje DLL pro OpenGL a wxWidgets – PyInstaller je zahrne automaticky
    • pylsl potřebuje liblsl.dll (Windows) → zkopírováno z site-packages/pylsl
    • Ujistěte se, že jste v aktivním .venv310 nebo conda prostředí s nainstalovanými závislostmi
    • Kompilace trvá 3-10 minut a výsledek může mít 300+ MB kvůli scipy/MNE/PsychoPy
"""

import subprocess
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv310" / "Scripts" / "python.exe"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def _find_python() -> str:
    """Najde Python interpreter pro build."""
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def _find_liblsl() -> list[tuple[str, str]]:
    """Najde liblsl.dll pro pylsl a vrátí ho jako bindata pro PyInstaller."""
    try:
        import pylsl
        pylsl_dir = Path(pylsl.__file__).parent
        dll_candidates = list(pylsl_dir.glob("*.dll")) + list(pylsl_dir.glob("lib/*.dll"))
        if dll_candidates:
            dll = dll_candidates[0]
            return [(str(dll), ".")]
    except ImportError:
        pass
    return []


def build():
    python = _find_python()
    liblsl_binaries = _find_liblsl()

    # Sestavit --add-binary argumenty pro liblsl
    binary_args = []
    for src, dst in liblsl_binaries:
        binary_args += ["--add-binary", f"{src}{';' if sys.platform == 'win32' else ':'}{dst}"]

    cmd = [
        python, "-m", "PyInstaller",
        "--name", "eeg_app",
        "--onedir",                    # složka (rychlejší start než --onefile)
        "--windowed",                  # bez konzole (okno)
        "--clean",
        "--noconfirm",
        # Ikona (pokud existuje)
        *( ["--icon", str(PROJECT_ROOT / "assets" / "icon.ico")]
           if (PROJECT_ROOT / "assets" / "icon.ico").exists() else [] ),
        # Zahrnout config a data složky
        "--add-data", f"{PROJECT_ROOT / 'config'};config",
        "--add-data", f"{PROJECT_ROOT / 'src'};src",
        # pylsl DLL
        *binary_args,
        # Skryté importy – MNE, psychopy a sklearn mohou mít dynamické importy
        "--hidden-import", "mne",
        "--hidden-import", "sklearn.utils._cython_blas",
        "--hidden-import", "sklearn.neighbors._typedefs",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "psychopy",
        "--hidden-import", "psychopy.visual",
        "--hidden-import", "psychopy.event",
        "--hidden-import", "psychopy.core",
        "--hidden-import", "pylsl",
        "--hidden-import", "joblib",
        "--hidden-import", "scipy.signal",
        "--hidden-import", "scipy.linalg",
        # Vyloučit velké nepoužívané balíky
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "numpy.testing",
        "--exclude-module", "pytest",
        "--exclude-module", "sphinx",
        # Entry point
        str(PROJECT_ROOT / "run_app.py"),
    ]

    print("=" * 60)
    print("EEG App – sestavení EXE")
    print("=" * 60)
    print(f"Python:  {python}")
    print(f"Výstup:  {DIST_DIR / 'eeg_app'}")
    if liblsl_binaries:
        print(f"liblsl:  {liblsl_binaries[0][0]}")
    else:
        print("liblsl:  NENALEZENO – ručně zkopírujte liblsl.dll do dist/eeg_app/")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("\n✗ Sestavení selhalo!")
        sys.exit(result.returncode)

    print("\n✓ Sestavení dokončeno!")
    print(f"  Výsledek: {DIST_DIR / 'eeg_app' / 'eeg_app.exe'}")
    print()
    print("Distribuční balík:")
    portable = DIST_DIR / "eeg_app_portable"
    if portable.exists():
        shutil.rmtree(portable)
    shutil.copytree(DIST_DIR / "eeg_app", portable)
    print(f"  {portable}")
    print()
    print("Poznámky:")
    print("  • Zkopírujte složku eeg_app_portable na cílový počítač")
    print("  • Spusťte eeg_app.exe (nevyžaduje Python)")
    print("  • config/ složka je v dist/eeg_app/config/ – lze upravovat")


if __name__ == "__main__":
    build()
