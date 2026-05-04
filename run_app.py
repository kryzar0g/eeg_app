#!/usr/bin/env python3
"""Spouštěcí skript pro EEG BCI aplikaci.

Lze spustit přímo::

    python run_app.py

nebo poklikáním na soubor v průzkumníku souborů (pokud je Python
nastaven jako výchozí interpret pro soubory *.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ujistíme se, že kořen projektu je na sys.path, aby fungovaly importy balíčku src.*
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from src.main import run_gui_mode  # noqa: E402

if __name__ == "__main__":
  run_gui_mode()
