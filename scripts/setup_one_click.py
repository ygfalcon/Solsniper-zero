#!/usr/bin/env python3
"""Perform a one-click setup and launch for SolHunter Zero."""

from __future__ import annotations

import importlib.resources as resources
import os
import sys
import subprocess
import shutil
import tomllib

from solhunter_zero.macos_setup import ensure_tools
import solhunter_zero.env_config as env_config
from solhunter_zero.paths import ROOT
from scripts import quick_setup
from solhunter_zero.bootstrap_utils import ensure_deps
from solhunter_zero import device
from solhunter_zero.logging_utils import log_startup


REQUIRED_CFG_KEYS = {
    "solana_rpc_url": "https://api.mainnet-beta.solana.com",
    "dex_base_url": "https://quote-api.jup.ag",
    "agents": ["simulation"],
    "agent_weights": {"simulation": 1.0},
}


def _validate_config(path: os.PathLike[str]) -> None:
    """Ensure the configuration file contains required settings."""
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)

    missing = [k for k in REQUIRED_CFG_KEYS if k not in cfg or not cfg[k]]
    if missing:
        print(
            f"Missing required keys in {path}: {', '.join(missing)}",
            file=sys.stderr,
        )
        example = (
            "solana_rpc_url = \"https://api.mainnet-beta.solana.com\"\n"
            "dex_base_url = \"https://quote-api.jup.ag\"\n"
            "agents = [\"simulation\"]\n\n"
            "[agent_weights]\n"
            "simulation = 1.0\n"
        )
        print("Example configuration:\n\n" + example, file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Execute the automated setup steps then run the autopilot."""
    ensure_tools(non_interactive=True)
    env_config.configure_environment(ROOT)
    quick_setup.main(["--auto", "--non-interactive"])
    cfg_path = getattr(quick_setup, "CONFIG_PATH", None)
    if cfg_path:
        _validate_config(cfg_path)
    ensure_deps(install_optional=True)

    if "PYTEST_CURRENT_TEST" not in os.environ:
        METAL_INDEX = (
            device.METAL_EXTRA_INDEX[1]
            if len(getattr(device, "METAL_EXTRA_INDEX", [])) > 1
            else "https://download.pytorch.org/whl/metal"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                ".[fastjson,fastcompress,msgpack]",
                "--extra-index-url",
                METAL_INDEX,
            ]
        )

    if shutil.which("cargo") and shutil.which("rustup"):
        try:
            subprocess.check_call(
                [
                    "cargo",
                    "build",
                    "--release",
                    "--features=parallel",
                    "--manifest-path",
                    "route_ffi/Cargo.toml",
                ],
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as exc:
            msg = "Failed to build route_ffi with parallel feature"
            print(f"{msg}: {exc}")
            log_startup(f"{msg}: {exc}")

    device.initialize_gpu()

    start_all = resources.files("scripts") / "start_all.py"
    cmd = [sys.executable, str(start_all), "autopilot"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":  # pragma: no cover
    main()
