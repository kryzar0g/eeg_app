#!/usr/bin/env python3
"""EEG BCI Application launcher."""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── KRITICKY pro PyInstaller + multiprocessing na Windows ────────────────────
# Musi byt volano PRED importem cehokoli jineho, jinak spawnovany
# proces (paradigma) znovu spusti cely EXE = nekonecna smycka / crash.
import multiprocessing
multiprocessing.freeze_support()

# ── Oprava stdout/stderr pro --windowed EXE (jsou None bez konzole) ──────────
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# ── Cesta k projektu: v EXE pouzit slozku s .exe, ne temp rozbaleni ─────────
if getattr(sys, "frozen", False):
    # PyInstaller EXE - projektovy koren je vedle .exe
    _ROOT = Path(sys.executable).resolve().parent
else:
    _ROOT = Path(__file__).resolve().parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging_config import setup_logging
from src.main import run_gui_mode

if __name__ == "__main__":
    setup_logging(log_dir=_ROOT / "logs", level="INFO")
    run_gui_mode()
