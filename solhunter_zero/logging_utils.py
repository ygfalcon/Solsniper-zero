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
