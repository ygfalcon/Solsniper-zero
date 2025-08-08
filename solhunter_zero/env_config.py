from __future__ import annotations

"""Helpers for configuring environment variables for SolHunter Zero."""

from pathlib import Path
import os
import shutil
import sys

import tomllib

from . import env
from .logging_utils import log_startup
from .config import ENV_VARS
from .env_defaults import DEFAULTS
from .paths import ROOT

__all__ = ["configure_environment", "report_env_changes"]


def _update_env_file(env_file: Path, updates: dict[str, str]) -> None:
    """Persist ``updates`` to *env_file* replacing existing values."""

    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    existing = {line.split("=", 1)[0]: i for i, line in enumerate(lines) if "=" in line}
    for key, value in updates.items():
        line = f"{key}={value}"
        if key in existing:
            lines[existing[key]] = line
        else:
            lines.append(line)
    text = "\n".join(lines) + ("\n" if lines else "")
    env_file.write_text(text, encoding="utf-8")


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
    if not env_file.exists():
        example_file = Path(root) / ".env.example"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        if example_file.exists():
            shutil.copy(example_file, env_file)
            log_startup(
                f"Created environment file {env_file} from {example_file}"
            )
        else:
            env_file.touch()
            log_startup(f"Created environment file {env_file}")
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
                value_str = (
                    str(val).lower() if isinstance(val, bool) else str(val)
                )
                os.environ[env_name] = value_str
                applied[env_name] = value_str
                missing_lines.append(f"{env_name}={value_str}\n")

    added_vars: dict[str, str] = {}
    if missing_lines:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.touch(exist_ok=True)
        with env_file.open("a", encoding="utf-8") as fh:
            fh.writelines(missing_lines)
        for line in missing_lines:
            name, _, value = line.partition("=")
            added_vars[name] = value.strip()
        log_startup(
            f"Updated environment file {env_file} with: "
            f"{', '.join(added_vars)}"
        )
        report_env_changes(added_vars, env_file)

    for key, value in DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]

    # Validate MEV bundle configuration. If bundles are enabled but Jito
    # credentials are missing, either prompt for the token or disable bundles
    # and persist the change to ``.env``.
    mev_enabled = os.environ.get("USE_MEV_BUNDLES", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    jito_auth = os.environ.get("JITO_AUTH", "").strip()
    if mev_enabled and not jito_auth:
        token = ""
        if sys.stdin.isatty():
            try:
                token = input(
                    "JITO_AUTH is required for MEV bundles. Enter token or leave blank to disable: "
                ).strip()
            except EOFError:
                token = ""
        if token:
            os.environ["JITO_AUTH"] = token
            applied["JITO_AUTH"] = token
            _update_env_file(env_file, {"JITO_AUTH": token})
            report_env_changes({"JITO_AUTH": token}, env_file)
        else:
            os.environ["USE_MEV_BUNDLES"] = "false"
            applied["USE_MEV_BUNDLES"] = "false"
            _update_env_file(env_file, {"USE_MEV_BUNDLES": "false"})
            report_env_changes({"USE_MEV_BUNDLES": "false"}, env_file)

    # GPU-related environment variables are configured exclusively via
    # :func:`device.initialize_gpu` during launcher startup to keep a single
    # source of truth.  ``configure_environment`` deliberately avoids calling
    # :func:`device.ensure_gpu_env`.

    for key, value in applied.items():
        log_startup(f"{key}: {value}")

    return applied


def report_env_changes(
    changes: dict[str, str], env_file: Path | None = None
) -> None:
    """Print environment variable *changes* for command-line feedback."""

    if not changes:
        return
    if env_file is not None:
        print(f"Updated environment file {env_file} with:")
    else:
        print("Environment variables applied:")
    for key, value in changes.items():
        print(f"{key}={value}")
