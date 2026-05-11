#!/usr/bin/env python3
"""EEG BCI Application launcher.

Run directly:
    python run_app.py

Or double-click if Python is set as default handler for .py files.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logging_config import setup_logging
from src.main import run_gui_mode

if __name__ == "__main__":
    setup_logging(log_dir=_ROOT / "logs", level="INFO")
    run_gui_mode()
