"""Shared file logger for the viewer UI modules."""
from __future__ import annotations

import logging
from pathlib import Path

_logger: logging.Logger | None = None
_log_path: Path | None = None


def get_viewer_logger(log_path: str | Path | None = None) -> logging.Logger:
    """Return the shared viewer logger, creating it on first call.

    On first call the log file is created at *log_path* (default:
    ``./viewer_errors.log`` relative to the current working directory).
    Subsequent calls return the same logger regardless of *log_path*.
    """
    global _logger, _log_path

    if _logger is not None:
        return _logger

    if log_path is None:
        log_path = Path.cwd() / "viewer_errors.log"

    _log_path = Path(log_path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cell_activity_labeler.viewer")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(_log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    _logger = logger
    return logger


def get_log_path() -> Path | None:
    """Return the path of the active log file, or None if not yet initialised."""
    return _log_path
