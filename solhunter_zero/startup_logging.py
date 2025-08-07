from __future__ import annotations

import logging
import sys
from pathlib import Path
from functools import wraps
from typing import Any, Callable, TypeVar

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "startup.log"

_logger = logging.getLogger("startup")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _logger.addHandler(fh)
    _logger.addHandler(sh)


def log_info(message: str) -> None:
    _logger.info(message)


def log_warning(message: str) -> None:
    _logger.warning(message)


def log_error(message: str, *, exc_info: bool = False) -> None:
    _logger.error(message, exc_info=exc_info)


F = TypeVar("F", bound=Callable[..., Any])


def capture_exceptions(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:  # pragma: no cover - unexpected failures
            _logger.exception("Unhandled exception")
            _logger.error("Startup failed: %s", exc)
            _logger.error("See startup.log for details.")
            return 1

    return wrapper  # type: ignore[misc]

