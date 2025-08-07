from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log_startup(message: str) -> None:
    """Append *message* to ``startup.log`` with a timestamp."""
    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")
    except OSError:
        pass


def rotate_log(path: Path, max_bytes: int) -> None:
    """Rotate or truncate *path* before writing new output.

    When ``path`` exists and its size exceeds ``max_bytes`` it is moved to a
    ``.1`` backup. Otherwise the file is truncated to start fresh.
    """

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
