from __future__ import annotations

import argparse
import logging
from enum import Enum

from .config import load_config
from .gui_app import run_gui
from .offline_analysis import run_offline_from_file
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    """Application operation modes."""
    RECORD = "record"    # Motor imagery paradigm with LSL markers
    OFFLINE = "offline"  # Offline analysis from EDF/BDF file
    ONLINE = "online"    # Real-time BCI with trained model
    GUI = "gui"          # Graphical interface


def run_record_mode() -> None:
    """Run motor imagery paradigm and stream LSL markers."""
    from .lsl_acquisition import create_streams
    from .stimuli.paradigm_base import MotorImageryParadigm
    
    config = load_config()
    streams = create_streams(config)
    
    paradigm = MotorImageryParadigm(config, streams.marker_outlet)
    paradigm.run()


def run_offline_mode(file_path: str) -> None:
    """Load EEG file and run offline training pipeline."""
    logger.info(f"Starting offline analysis on {file_path}")
    acc, n_epochs = run_offline_from_file(file_path)
    logger.info(f"Offline analysis complete: {n_epochs} epochs, accuracy: {acc:.4f}")


def run_online_mode() -> None:
    """Run real-time BCI loop with trained model."""
    from .online_bci import run_online_bci
    
    config = load_config()
    run_online_bci(config)


def run_gui_mode() -> None:
    """Launch graphical interface."""
    from .gui_app_v2 import run_gui
    
    run_gui()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EEG Motor Imagery BCI Application")
    parser.add_argument(
        "mode",
        type=str,
        nargs="?",
        choices=[m.value for m in Mode],
        default=Mode.GUI.value,
        help="Operation mode (default: gui)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to EDF/BDF file for offline analysis",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger.info(f"Starting application in {args.mode} mode")
    
    try:
        mode = Mode(args.mode)
        
        if mode is Mode.RECORD:
            run_record_mode()
        elif mode is Mode.OFFLINE:
            if not args.file:
                raise SystemExit("Offline mode requires --file argument")
            run_offline_mode(args.file)
        elif mode is Mode.ONLINE:
            run_online_mode()
        elif mode is Mode.GUI:
            run_gui_mode()
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
