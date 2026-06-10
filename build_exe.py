"""
Sestaví standalone EXE soubor pomocí PyInstaller.

Spuštění:
    python build_exe.py

Výstup:
    dist/eeg_app/eeg_app.exe   (složka s EXE a všemi závislostmi)
    dist/eeg_app_portable/     (to samé, přejmenováno pro distribuci)

Poznámky:
    • MNE používá lazy_loader – .pyi stub soubory musí být v bundlu
      (--collect-data mne toto řeší automaticky)
    • PsychoPy vyžaduje DLL pro OpenGL/wxWidgets – zahrnuty automaticky
    • pylsl potřebuje liblsl.dll (Windows) → zkopírováno z site-packages/pylsl
    • Kompilace trvá 5-15 minut, výsledek ~400–600 MB kvůli MNE/PsychoPy/scipy
"""

import subprocess
import sys
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON  = PROJECT_ROOT / ".venv310" / "Scripts" / "python.exe"
DIST_DIR     = PROJECT_ROOT / "dist"
SEP          = ";" if sys.platform == "win32" else ":"


def _has_all_deps(python_exe: str) -> bool:
    """Overi, ze dany Python ma vsechny klicove zavislosti pro EXE."""
    import subprocess
    check = (
        "import pydantic, mne, edfio, numpy, scipy, sklearn, pylsl, yaml; "
        "print('OK')"
    )
    try:
        r = subprocess.run([python_exe, "-c", check],
                           capture_output=True, text=True, timeout=60)
        return "OK" in r.stdout
    except Exception:
        return False


def _find_python() -> str:
    """Vybere Python s KOMPLETNIMI zavislostmi.

    Preferuje interpret, ktery spustil tento skript (sys.executable),
    pokud ma vsechny balicky. Jinak zkusi .venv310. Tim se zabrani
    sestaveni z neuplneho prostredi (chybejici pydantic/edfio).
    """
    candidates = [sys.executable]
    if VENV_PYTHON.is_file():
        candidates.append(str(VENV_PYTHON))

    for py in candidates:
        if _has_all_deps(py):
            print(f"Pouzivam Python s kompletnimi zavislostmi: {py}")
            return py

    # Nikdo nema vse - varovat a pouzit sys.executable
    print("VAROVANI: zadny interpret nema vsechny zavislosti!")
    print("  Nainstalujte: python -m pip install -r requirements.txt")
    return sys.executable


def _find_liblsl() -> list:
    """Najde liblsl.dll pro pylsl."""
    try:
        import pylsl
        pylsl_dir = Path(pylsl.__file__).parent
        for pattern in ("*.dll", "lib/*.dll", "**/*.dll"):
            hits = list(pylsl_dir.glob(pattern))
            if hits:
                return [(str(hits[0]), ".")]
    except ImportError:
        pass
    return []


def _mne_pyi_data() -> list:
    """Vrátí --add-data argument pro mne/__init__.pyi stub soubor.

    MNE lazy_loader vyžaduje tento soubor přímo vedle mne/__init__.py.
    Pokud stub neexistuje (starší MNE), vrátí prázdný seznam.
    """
    try:
        import mne
        pyi = Path(mne.__file__).parent / "__init__.pyi"
        if pyi.exists():
            return [(str(pyi), "mne")]
    except ImportError:
        pass
    return []


def build():
    python         = _find_python()
    liblsl_bins    = _find_liblsl()
    mne_pyi        = _mne_pyi_data()

    sep = SEP

    # --add-binary pro liblsl
    binary_args = []
    for src, dst in liblsl_bins:
        binary_args += ["--add-binary", f"{src}{sep}{dst}"]

    # --add-data pro mne .pyi stub (fix lazy_loader ValueError)
    pyi_args = []
    for src, dst in mne_pyi:
        pyi_args += ["--add-data", f"{src}{sep}{dst}"]

    cmd = [
        python, "-m", "PyInstaller",
        "--name", "eeg_app",
        "--onedir",       # složka = rychlejší start než --onefile
        "--windowed",     # bez konzole (GUI aplikace)
        "--clean",
        "--noconfirm",

        # Ikona (pokud existuje)
        *(["--icon", str(PROJECT_ROOT / "assets" / "icon.ico")]
          if (PROJECT_ROOT / "assets" / "icon.ico").exists() else []),

        # ── Data (config/ musi byt VEDLE .exe, ne uvnitr _internal/) ──
        # run_app.py pouziva sys.executable.parent jako PROJECT_ROOT,
        # takze config/ musi byt primo v dist/eeg_app/ (ne v _internal/).
        "--add-data", f"{PROJECT_ROOT / 'config'}{sep}config",
        "--add-data", f"{PROJECT_ROOT / 'src'}{sep}src",

        # MNE .pyi stub – oprava "Cannot load imports from non-existent stub"
        *pyi_args,

        # --collect-data zahrne VSE datové soubory balíku (včetně .pyi stubů)
        # Toto je nejspolehlivější oprava pro mne + lazy_loader
        "--collect-data", "mne",
        "--collect-data", "lazy_loader",

        # pylsl DLL
        *binary_args,

        # ── Skryté importy ───────────────────────────────────────────
        "--hidden-import", "mne",
        "--hidden-import", "mne.io",
        "--hidden-import", "mne.io.edf",
        "--hidden-import", "mne.export",
        "--hidden-import", "edfio",
        "--hidden-import", "sklearn.utils._cython_blas",
        "--hidden-import", "sklearn.neighbors._typedefs",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "sklearn.discriminant_analysis",
        "--hidden-import", "sklearn.svm",
        "--hidden-import", "psychopy",
        "--hidden-import", "psychopy.visual",
        "--hidden-import", "psychopy.event",
        "--hidden-import", "psychopy.core",
        "--hidden-import", "pylsl",
        "--hidden-import", "joblib",
        "--hidden-import", "scipy.signal",
        "--hidden-import", "scipy.linalg",
        "--hidden-import", "scipy.sparse.csgraph._validation",
        "--hidden-import", "pydantic",
        "--hidden-import", "yaml",
        # ERD analyza + grafy
        "--hidden-import", "src.analysis_erd",
        "--hidden-import", "src.lsl_network",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.backends.backend_agg",
        "--collect-data", "matplotlib",

        # ── multiprocessing freeze_support (KRIT. pro Windows EXE) ──
        # Bez tohoto spawnovany proces (paradigma) znovu spusti
        # cely EXE jako modul = nekonecna smycka / crash.
        "--hidden-import", "multiprocessing.spawn",
        "--hidden-import", "multiprocessing.forkserver",

        # ── Vyloučit nepotřebné velké balíky ────────────────────────
        # POZOR: numpy.testing NESMI byt vylouceno - scikit-learn ho potrebuje!
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "pytest",
        "--exclude-module", "sphinx",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",

        # Entry point
        str(PROJECT_ROOT / "run_app.py"),
    ]

    print("=" * 60)
    print("EEG App – sestaveni EXE")
    print("=" * 60)
    print(f"Python : {python}")
    print(f"Vystup : {DIST_DIR / 'eeg_app'}")
    print(f"liblsl : {liblsl_bins[0][0] if liblsl_bins else 'NENALEZENO'}")
    print(f"mne.pyi: {mne_pyi[0][0] if mne_pyi else 'nenalezeno (collect-data MNE pokryje)'}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("\nSesteveni selhalo! Viz vystup nahore.")
        sys.exit(result.returncode)

    exe_dir = DIST_DIR / "eeg_app"

    # ── Post-build: zkopirovat config/ VEDLE .exe (ne do _internal/) ──
    # run_app.py pouziva sys.executable.parent jako PROJECT_ROOT v EXE,
    # takze config/ musi byt primo v dist/eeg_app/.
    dst_config = exe_dir / "config"
    if dst_config.exists():
        shutil.rmtree(dst_config)
    shutil.copytree(PROJECT_ROOT / "config", dst_config)
    print(f"  config/ zkopirovan do: {dst_config}")

    # Vytvorit prazdne slozky pro data a modely
    for folder in ("data/recordings", "models", "logs"):
        (exe_dir / folder).mkdir(parents=True, exist_ok=True)

    print("\nSesteveni dokonceno!")
    print(f"  EXE: {exe_dir / 'eeg_app.exe'}")

    portable = DIST_DIR / "eeg_app_portable"
    if portable.exists():
        shutil.rmtree(portable)
    shutil.copytree(exe_dir, portable)
    print(f"  Prenosny balik: {portable}")
    print()
    print("Pouziti:")
    print("  Zkopirujte slozku eeg_app_portable na cilovy pocitac")
    print("  Spustte eeg_app.exe (nevyzaduje Python)")
    print("  config/ slozka je vedle .exe – lze upravovat")


if __name__ == "__main__":
    build()
