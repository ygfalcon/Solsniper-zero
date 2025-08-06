from __future__ import annotations

"""Runtime environment preparation helpers.

This module centralizes startup logic so that all entry points can ensure the
runtime environment is ready before executing the main application.  The
``prepare_environment`` function mirrors the setup performed by the interactive
``scripts/startup.py`` helper but is safe to call programmatically.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``path`` into ``os.environ``.

    The parser is intentionally small and ignores blank lines and comments.  If
    a variable is already defined in the environment its value is preserved.
    """

    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def prepare_environment(one_click: bool = False) -> None:
    """Prepare the runtime environment for SolHunter Zero.

    Parameters
    ----------
    one_click:
        When ``True`` the helper enables fully automated non-interactive setup
        such as automatically selecting a default keypair when none is active.
    """

    # Load environment variables from optional .env file and ensure we execute
    # from the project root.
    _load_env_file(ROOT / ".env")
    os.chdir(ROOT)
    os.environ.setdefault("DEPTH_SERVICE", "true")

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    # Delegate the heavy lifting to the existing bootstrap helper which handles
    # virtualenv management, dependency installation, configuration creation and
    # Rust builds for the FFI components.
    from .bootstrap import bootstrap

    bootstrap(one_click=one_click)

    # Configure GPU related environment and default PyTorch device.
    from . import device

    device.ensure_gpu_env()
    try:  # torch is optional in minimal installs
        import torch

        torch.set_default_device(device.get_default_device())
    except Exception:  # pragma: no cover - torch may be absent
        pass

    gpu_available = device.detect_gpu()
    gpu_device = str(device.get_default_device()) if gpu_available else "none"
    os.environ["SOLHUNTER_GPU_AVAILABLE"] = "1" if gpu_available else "0"
    os.environ["SOLHUNTER_GPU_DEVICE"] = gpu_device
