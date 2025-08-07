from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from solhunter_zero.bootstrap_utils import (
    ensure_cargo,
    ensure_deps,
    ensure_depth_service,
    ensure_route_ffi,
    ensure_venv,
)
from scripts.startup import ensure_keypair
from .config_bootstrap import ensure_config
from . import wallet
from . import env

import solhunter_zero.device as device

ROOT = Path(__file__).resolve().parent.parent


def verify_launch_ready() -> tuple[bool, str]:
    """Check that required files and tools exist before launching.

    Returns a tuple ``(ok, message)`` where ``ok`` indicates whether all
    prerequisites are satisfied. The message contains a human-readable
    description of the first missing requirement or a success note.
    """

    config_path = ROOT / "config.toml"
    if not config_path.exists():
        return False, "Missing config.toml. Run scripts/startup.py or scripts/quick_setup.py."

    keypair_dir = ROOT / "keypairs"
    active = keypair_dir / "active"
    if not active.exists():
        return False, "No active keypair found. Run scripts/startup.py to create one."
    name = active.read_text().strip()
    keyfile = keypair_dir / f"{name}.json"
    if not keyfile.exists():
        return False, f"Active keypair '{name}.json' not found. Run scripts/startup.py."

    if shutil.which("cargo") is None:
        return False, "Rust toolchain (cargo) not found. Run scripts/startup.py --full-deps."

    libname = "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    libpath = ROOT / "solhunter_zero" / libname
    if not libpath.exists():
        return False, f"{libname} missing. Run scripts/startup.py to build it."

    depth_bin = ROOT / "target" / "release" / "depth_service"
    if not depth_bin.exists():
        return False, "depth_service binary missing. Run scripts/startup.py to build it."

    return True, "All prerequisites satisfied."


def bootstrap(one_click: bool = False) -> None:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically. It automatically loads the project's ``.env`` file,
    making it self-contained regarding environment setup.
    """
    env.load_env_file(ROOT / ".env")
    device.ensure_gpu_env()

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if os.getenv("SOLHUNTER_SKIP_VENV") != "1":
        ensure_venv(None)

    if os.getenv("SOLHUNTER_SKIP_DEPS") != "1":
        ensure_deps(
            install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1"
        )

    if os.getenv("SOLHUNTER_SKIP_SETUP") != "1":
        ensure_config()
        ensure_keypair()

    wallet.ensure_default_keypair()
    ensure_cargo()
    ensure_route_ffi()
    ensure_depth_service()
