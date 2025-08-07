from __future__ import annotations

"""Helpers for configuring environment variables for SolHunter Zero."""

from pathlib import Path
import os

import tomllib

from . import env
from .logging_utils import log_startup
from .config import ENV_VARS
from .env_defaults import DEFAULTS
from .paths import ROOT

__all__ = ["configure_environment"]


def configure_environment(root: Path | None = None) -> dict[str, str]:
    """Load ``.env`` and apply defaults defined in :mod:`env_defaults`.

    GPU environment variables are intentionally not configured here. They are
    handled by :func:`solhunter_zero.device.initialize_gpu` during the launcher
    startup sequence to maintain a single source of truth.

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
    env_file = Path(root) / ".env"
    env.load_env_file(env_file)

    applied: dict[str, str] = {}
    missing_lines: list[str] = []

    cfg_path = Path(root) / "config.toml"
    if cfg_path.exists():
        try:
            with cfg_path.open("rb") as fh:
                cfg = tomllib.load(fh)
        except Exception:
            cfg = {}
        for key, env_name in ENV_VARS.items():
            val = cfg.get(key)
            if val is not None and env_name not in os.environ:
                value_str = str(val).lower() if isinstance(val, bool) else str(val)
                os.environ[env_name] = value_str
                applied[env_name] = value_str
                missing_lines.append(f"{env_name}={value_str}\n")

    if missing_lines:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.touch(exist_ok=True)
        with env_file.open("a", encoding="utf-8") as fh:
            fh.writelines(missing_lines)

    for key, value in DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]

    # GPU-related environment variables are configured exclusively via
    # :func:`device.initialize_gpu` during launcher startup to keep a single
    # source of truth.  ``configure_environment`` deliberately avoids calling
    # :func:`device.ensure_gpu_env`.

    for key, value in applied.items():
        log_startup(f"{key}: {value}")

    return applied

