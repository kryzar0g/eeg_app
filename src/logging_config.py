import io
import logging
import logging.handlers
import sys
from pathlib import Path

__all__ = ["get_logger", "setup_logging"]


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler ktery nikdy nespadne na UnicodeEncodeError.

    Na Windows s cp1252 konzoli nahrazuje neznake znaky '?'
    misto vyhazovani vyjimky ktera by prerusila vlakno.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback: zakodovat s nahradou neznakych znaku
                safe = (msg + self.terminator).encode(
                    getattr(stream, "encoding", "utf-8") or "utf-8",
                    errors="replace",
                ).decode(
                    getattr(stream, "encoding", "utf-8") or "utf-8",
                    errors="replace",
                )
                stream.write(safe)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> Path:
    """Configure application-wide logging with Unicode-safe console output."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Odstranit existujici handlery (vcetne Python lastResort s cp1252)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (UTF-8, rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "eeg_app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler – bezpecny vuci cp1252
    console_handler = _SafeStreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return log_dir


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)
