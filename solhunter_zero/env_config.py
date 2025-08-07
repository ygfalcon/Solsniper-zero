from __future__ import annotations

"""Helpers for configuring environment variables for SolHunter Zero."""

from pathlib import Path
import os

from . import env, device
from .logging_utils import log_startup

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["configure_environment"]


_DEFAULTS: dict[str, str] = {
    "DEPTH_SERVICE": "true",
}


def configure_environment(root: Path | None = None) -> dict[str, str]:
    """Load ``.env`` and ensure default and GPU variables are set.

    Parameters
    ----------
    root:
        Optional project root path.  Defaults to the repository root.

    Returns
    -------
    dict[str, str]
        Mapping of variables that were applied.
    """

    root = root or ROOT
    env.load_env_file(Path(root) / ".env")

    applied: dict[str, str] = {}
    for key, value in _DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]

    gpu_env = device.ensure_gpu_env()
    applied.update(gpu_env)

    for key, value in applied.items():
        log_startup(f"{key}: {value}")

    return applied

