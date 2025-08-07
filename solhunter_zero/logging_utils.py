from __future__ import annotations

import logging
import sys
from pathlib import Path

from .paths import ROOT

MAX_STARTUP_LOG_SIZE = 1_000_000  # 1 MB
MAX_PREFLIGHT_LOG_SIZE = 1_000_000  # 1 MB


def rotate_startup_log(path: Path = ROOT / "startup.log") -> None:
    """Rotate or truncate the startup log before writing new output."""

    if not path.exists():
        return
    try:
        if path.stat().st_size > MAX_STARTUP_LOG_SIZE:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError:
        pass


def rotate_preflight_log(
    path: Path = ROOT / "preflight.log", max_bytes: int = MAX_PREFLIGHT_LOG_SIZE
) -> None:
    """Rotate or truncate the preflight log before writing new output."""

    if not path.exists():
        return
    try:
        if path.stat().st_size > max_bytes:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError:
        pass


def init_startup_logging() -> logging.Logger:
    """Initialize startup logging and return the configured logger.

    The function rotates both ``startup.log`` and ``preflight.log`` then
    configures and returns a :class:`logging.Logger` that writes timestamped
    messages to ``startup.log`` and the console.
    """

    rotate_startup_log()
    rotate_preflight_log()

    logger = logging.getLogger("startup")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%dT%H:%M:%S")

    file_handler = logging.FileHandler(ROOT / "startup.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def log_startup(message: str) -> None:
    """Append *message* to ``startup.log`` with a timestamp."""

    logging.getLogger("startup").info(message)
