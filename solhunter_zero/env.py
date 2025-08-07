from __future__ import annotations

"""Utilities for loading environment variables from files."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["load_env_file"]


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from *path* into ``os.environ``.

    Blank lines and ``#`` comments are ignored. Existing environment variables
    are preserved. Missing files are created (empty or copied from a ``.example``
    template) and noted in ``startup.log``.
    """

    if not path.exists():
        template = (
            path.with_suffix(path.suffix + ".example")
            if path.suffix
            else path.with_name(path.name + ".example")
        )
        data = template.read_text() if template.exists() else ""
        try:
            path.write_text(data)
            os.chmod(path, 0o600)
            msg = f"Created environment file {path}"
        except OSError as exc:
            msg = f"Warning: environment file {path} could not be created: {exc}"
            print(msg)
            try:
                with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass
            return
        print(msg)
        try:
            with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
