import io
import logging
import logging.handlers
import sys
from pathlib import Path
import os

__all__ = ["get_logger", "setup_logging"]


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> Path:
    """Configure application-wide logging.
    
    Args:
        log_dir: Directory for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        Path to log directory.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    if root_logger.handlers:
        return log_dir
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "eeg_app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler – UTF-8 s fallbackem na '?' pro znaky mimo cp1252
    try:
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        console_handler = logging.StreamHandler(stream=utf8_stream)
    except AttributeError:
        # stdout nema .buffer (napr. IDLE, redirected) – pouzit errors=replace
        console_handler = logging.StreamHandler()
        console_handler.stream.errors = "replace"  # type: ignore[attr-defined]
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_dir


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)
