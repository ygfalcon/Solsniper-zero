from __future__ import annotations

import platform
import subprocess
import shutil
from pathlib import Path


def _rust_version(cmd: str) -> str:
    if shutil.which(cmd) is None:
        return "not installed"
    try:
        return subprocess.check_output([cmd, "--version"], text=True).strip()
    except Exception:
        return "error"


def collect() -> dict[str, str]:
    info: dict[str, str] = {}
    info["python"] = platform.python_version()

    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        info["torch"] = "not installed"

    try:
        from solhunter_zero import device

        info["gpu_backend"] = device.get_gpu_backend() or "none"
    except Exception:
        info["gpu_backend"] = "unknown"

    info["rustc"] = _rust_version("rustc")
    info["cargo"] = _rust_version("cargo")

    config_present = any(
        Path(name).is_file() for name in ("config.toml", "config.yaml", "config.yml")
    )
    info["config"] = "present" if config_present else "missing"

    try:
        from solhunter_zero import wallet

        keypairs = wallet.list_keypairs()
        active = wallet.get_active_keypair_name()
        if active:
            info["keypair"] = active
        elif keypairs:
            info["keypair"] = keypairs[0]
        else:
            info["keypair"] = "missing"
    except Exception:
        info["keypair"] = "error"

    return info


def main() -> int:
    info = collect()
    for key, val in info.items():
        print(f"{key}: {val}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
