from __future__ import annotations

import shutil  # used for copying template configs
import subprocess
import sys
from pathlib import Path

import tomllib

__all__ = ["ensure_config"]

from .paths import ROOT


def _copy_template(dst: Path) -> bool:
    """Copy the default config template to ``dst``.

    Returns ``True`` if the template file was copied.
    """
    src = ROOT / "config" / "default.toml"
    if src.exists():
        shutil.copy(src, dst)
        return True
    return False


def _ensure_tomli_w():
    """Import ``tomli_w`` installing it via pip if necessary."""
    try:
        import tomli_w  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli-w"])
            import tomli_w  # type: ignore
        except Exception as exc:  # pragma: no cover - installation failure
            print(f"Failed to install 'tomli-w': {exc}")
            raise SystemExit(1)
    return tomli_w


def ensure_config(cfg_path: str | Path | None = None) -> Path:
    """Ensure ``cfg_path`` exists and contains valid configuration.

    If the configuration file is missing a default template is copied. Environment
    overrides are applied and the resulting configuration is validated and written
    back to disk. The final path to the configuration file is returned.
    """
    from .config import apply_env_overrides, validate_config

    cfg_file = Path(cfg_path) if cfg_path is not None else ROOT / "config.toml"
    created = False
    if not cfg_file.exists():
        created = _copy_template(cfg_file)

    tomli_w = _ensure_tomli_w()

    if cfg_file.exists():
        with cfg_file.open("rb") as fh:
            cfg = tomllib.load(fh)
    else:
        cfg = {}

    cfg = apply_env_overrides(cfg)
    try:
        cfg = validate_config(cfg)
    except ValueError as exc:  # pragma: no cover - config validation
        print(f"Invalid configuration: {exc}")
        raise SystemExit(1)

    with cfg_file.open("wb") as fh:
        fh.write(tomli_w.dumps(cfg).encode("utf-8"))
    if created:
        print(f"Configuration created at {cfg_file}")
    return cfg_file
