from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def rotate_log(path: Path | str, max_bytes: int) -> None:
    """Rotate or truncate the log at *path*.

    When ``path`` exists and exceeds ``max_bytes`` it is moved to ``.1``.
    Otherwise the file is truncated to start fresh for the current run.
    """

    path = Path(path)
    if not path.exists():
        return
    try:
        if path.stat().st_size > max_bytes:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError as exc:
        print(f"Failed to rotate log {path}: {exc}", file=sys.stderr)


def log_startup(message: str) -> None:
    """Append *message* to ``startup.log`` with a timestamp."""
    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")
    except OSError:
        pass
