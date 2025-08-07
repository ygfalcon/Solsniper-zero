from __future__ import annotations

from datetime import datetime
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


def log_startup(message: str) -> None:
    """Append *message* to ``startup.log`` with a timestamp."""
    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")
    except OSError:
        pass
