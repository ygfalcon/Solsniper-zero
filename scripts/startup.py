#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""

from __future__ import annotations

import sys

import argparse
import os
import platform
import subprocess
import shutil
import time
from pathlib import Path

from scripts import deps
from scripts.rust_utils import build_depth_service, build_route_ffi, ensure_cargo

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEPTH_SERVICE", "true")

if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from solhunter_zero import device

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore

if torch:
    os.environ.setdefault("TORCH_DEVICE", str(device.get_default_device()))


def ensure_venv(argv: list[str] | None) -> None:
    """Create a local virtual environment and re-invoke the script inside it.

    The environment is created in ``ROOT/.venv`` using ``python -m venv``.  If
    the script is not already executing from that interpreter, ``execv`` is used
    to restart the current process with ``.venv/bin/python`` (or the Windows
    equivalent).  When ``argv`` is not ``None`` the function assumes it is being
    called from tests and does nothing to avoid spawning subprocesses.  If the
    environment exists but the interpreter is missing or not executable the
    directory is removed and recreated before continuing.
    """

    if argv is not None:  # avoid side effects during tests
        return

    venv_dir = ROOT / ".venv"
    python = (
        venv_dir
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )

    if venv_dir.exists():
        if not python.exists() or not os.access(str(python), os.X_OK):
            print("Virtual environment missing interpreter; recreating .venv...")
            shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print("Creating virtual environment in .venv...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"Failed to create .venv: {exc}")
            raise SystemExit(1)

    if Path(sys.prefix) != venv_dir:
        os.execv(str(python), [str(python), *sys.argv])


def _pip_install(*args: str, retries: int = 3) -> None:
    """Run ``pip install`` with retries and exponential backoff."""
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            wait = 2 ** (attempt - 1)
            print(
                f"pip install {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    print(f"Failed to install {' '.join(args)} after {retries} attempts:")
    for err in errors:
        if err:
            print(err)
    raise SystemExit(result.returncode)


def ensure_deps(*, install_optional: bool = False) -> None:
    req, opt = deps.check_deps()
    if not req and not install_optional:
        if opt:
            print(
                "Optional modules missing: "
                + ", ".join(opt)
                + " (features disabled)."
            )
        return

    # Ensure ``pip`` is available before attempting installations.
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        if "No module named pip" in pip_check.stderr:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "ensurepip", "--default-pip"]
                )
            except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
                print(f"Failed to bootstrap pip: {exc}")
                raise SystemExit(exc.returncode)
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        else:  # pragma: no cover - unexpected failure
            print(f"Failed to invoke pip: {pip_check.stderr.strip()}")
            raise SystemExit(pip_check.returncode)

    if req:
        print("Installing required dependencies...")
        _pip_install(".[uvloop]")

    if install_optional and "torch" in opt and platform.system() == "Darwin" and platform.machine() == "arm64":
        print(
            "Installing torch==2.1.0 and torchvision==0.16.0 for macOS arm64 with Metal support..."
        )
        _pip_install(
            "torch==2.1.0",
            "torchvision==0.16.0",
            "--extra-index-url",
            "https://download.pytorch.org/whl/metal",
        )
        opt.remove("torch")

    if install_optional and platform.system() == "Darwin" and platform.machine() == "arm64":
        import importlib
        import torch

        if not torch.backends.mps.is_available():
            print(
                "MPS backend not available; attempting to reinstall Metal wheel..."
            )
            try:
                _pip_install(
                    "--force-reinstall",
                    "torch==2.1.0",
                    "torchvision==0.16.0",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/metal",
                )
            except SystemExit:
                print("Failed to reinstall torch with Metal wheels.")
            importlib.reload(torch)
            if not torch.backends.mps.is_available():
                raise SystemExit(
                    "MPS backend still not available. Please install the Metal wheel manually:\n"
                    "pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url "
                    "https://download.pytorch.org/whl/metal"
                )

    if install_optional and opt:
        print("Installing optional dependencies...")
        mapping = {
            "faiss": "faiss-cpu",
            "sentence_transformers": "sentence-transformers",
            "torch": "torch",
            "orjson": "orjson",
            "lz4": "lz4",
            "zstandard": "zstandard",
            "msgpack": "msgpack",
        }
        mods = set(opt)
        extras: list[str] = []
        if "orjson" in mods:
            extras.append("fastjson")
        if {"lz4", "zstandard"} & mods:
            extras.append("fastcompress")
        if "msgpack" in mods:
            extras.append("msgpack")
        if extras:
            _pip_install(f".[{','.join(extras)}]")
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            _pip_install(pkg)

    req_after, opt_after = deps.check_deps()
    missing_opt = list(opt_after)
    if req_after:
        print(
            "Missing required dependencies after installation: "
            + ", ".join(req_after)
        )
    if missing_opt:
        print(
            "Optional modules missing: "
            + ", ".join(missing_opt)
            + " (features disabled)."
        )
    if req_after:
        raise SystemExit(1)


def ensure_config() -> None:
    if not any(Path(name).is_file() for name in ("config.toml", "config.yaml", "config.yml")):
        from scripts import quick_setup

        quick_setup.main(["--auto"])


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available.

    If the command is missing, attempt to install the current package which
    provides it. On failure, instruct the user and abort gracefully.
    """
    if shutil.which("solhunter-wallet") is not None:
        return

    print("'solhunter-wallet' command not found. Installing the package...")
    try:
        _pip_install(".")
    except SystemExit:
        print("Failed to install 'solhunter-wallet'. Please run 'pip install .' manually.")
        raise

    if shutil.which("solhunter-wallet") is None:
        print("'solhunter-wallet' still not available after installation. Aborting.")
        raise SystemExit(1)


def ensure_default_keypair() -> None:
    """Create and select a default keypair if none is active.

    The helper checks for ``keypairs/active`` and, when missing, invokes the
    ``scripts/setup_default_keypair.sh`` helper via ``subprocess``.  The shell
    script prints the mnemonic only when it generates one.  In that case the
    mnemonic is echoed here with guidance to store it securely.
    """

    active_file = ROOT / "keypairs" / "active"
    if active_file.exists():
        return

    setup_script = ROOT / "scripts" / "setup_default_keypair.sh"
    try:
        result = subprocess.run(
            ["bash", str(setup_script)], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Failed to run {setup_script}: {exc}")
        raise SystemExit(1)

    mnemonic = result.stdout.strip()
    if mnemonic:
        print(f"Generated mnemonic: {mnemonic}")
        print("Please store this mnemonic securely; it will not be shown again.")


def ensure_keypair() -> None:
    from solhunter_zero import wallet
    from bip_utils import Bip39MnemonicGenerator

    def _wallet_cmd(*args: str) -> list[str]:
        if shutil.which("solhunter-wallet") is not None:
            return ["solhunter-wallet", *args]
        return [sys.executable, "-m", "solhunter_zero.wallet_cli", *args]

    keypairs = wallet.list_keypairs()
    active = wallet.get_active_keypair_name()
    if keypairs:
        if len(keypairs) == 1 and active is None:
            name = keypairs[0]
            wallet.select_keypair(name)
            print(f"Automatically selected keypair '{name}'.")
        return

    mnemonic = os.environ.get("MNEMONIC")
    keypair_json = os.environ.get("KEYPAIR_JSON")
    if not mnemonic and not keypair_json:
        mnemonic = str(Bip39MnemonicGenerator().FromWordsNumber(24))
        passphrase = os.environ.get("PASSPHRASE", "")
        subprocess.check_call(
            _wallet_cmd("derive", "default", mnemonic, "--passphrase", passphrase)
        )
        subprocess.check_call(_wallet_cmd("select", "default"))
        os.environ.setdefault("MNEMONIC", mnemonic)
        name = wallet.get_active_keypair_name() or "default"
        print(f"Generated mnemonic: {mnemonic}")
        print("Please store this mnemonic securely; it will not be shown again.")
        print(f"Automatically generated keypair '{name}' and selected it.")
        return

    print("No keypairs found in 'keypairs/' directory.")
    if keypair_json:
        subprocess.check_call(_wallet_cmd("save", "default", keypair_json))
        subprocess.check_call(_wallet_cmd("select", "default"))
        print("Keypair saved from KEYPAIR_JSON and selected as 'default'.")
        return

    if mnemonic:
        passphrase = os.environ.get("PASSPHRASE", "")
        subprocess.check_call(
            _wallet_cmd("derive", "default", mnemonic, "--passphrase", passphrase)
        )
        subprocess.check_call(_wallet_cmd("select", "default"))
        print("Keypair saved from MNEMONIC and selected as 'default'.")
        return


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints.  This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable.  BirdEye is only checked when an API key is configured.
    """

    import urllib.error
    import urllib.request
    import time

    urls: dict[str, str] = {}
    if cfg.get("birdeye_api_key"):
        urls["BirdEye"] = "https://public-api.birdeye.so/defi/tokenlist"
    for key, val in cfg.items():
        if not isinstance(val, str):
            continue
        if not val.startswith("http://") and not val.startswith("https://"):
            continue
        urls[key] = val

    failed: list[str] = []
    for name, url in urls.items():
        req = urllib.request.Request(url, method="HEAD")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=5):  # nosec B310
                    break
            except urllib.error.URLError as exc:  # pragma: no cover - network failure
                if attempt == 2:
                    print(
                        f"Failed to reach {name} at {url} after 3 attempts: {exc}."
                        " Check your network connection or configuration."
                    )
                    failed.append(name)
                else:
                    wait = 2**attempt
                    print(
                        f"Attempt {attempt + 1} failed for {name} at {url}: {exc}."
                        f" Retrying in {wait} seconds..."
                    )
                    time.sleep(wait)

    if failed:
        raise SystemExit(1)


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
    rpc_url = os.environ.get(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )
    if not os.environ.get("SOLANA_RPC_URL"):
        print(f"Using default RPC URL {rpc_url}")

    import json
    import urllib.request
    import time

    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getHealth"}).encode()
    req = urllib.request.Request(
        rpc_url, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                resp.read()
                break
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                msg = (
                    f"Failed to contact Solana RPC at {rpc_url} after 3 attempts: {exc}."
                    " Please ensure the endpoint is reachable or set SOLANA_RPC_URL to a valid RPC."
                )
                if warn_only:
                    print(f"Warning: {msg}")
                    return
                print(msg)
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}.",
                f" Retrying in {wait} seconds...",
            )
            time.sleep(wait)

def main(argv: list[str] | None = None) -> int:
    ensure_venv(argv)

    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument(
        "--full-deps",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip config and wallet prompts")
    parser.add_argument(
        "--skip-rpc-check", action="store_true", help="Skip Solana RPC availability check"
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip HTTP endpoint availability check",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip environment preflight checks",
    )
    parser.add_argument(
        "--one-click",
        action="store_true",
        help="Enable fully automated non-interactive startup",
    )
    parser.add_argument(
        "--allow-rosetta",
        action="store_true",
        help="Allow running under Rosetta (no Metal acceleration)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print system diagnostics and exit",
    )
    args, rest = parser.parse_known_args(argv)

    if args.diagnostics:
        from scripts import diagnostics

        diagnostics.main()
        return 0
    if args.one_click:
        rest = ["--non-interactive", *rest]
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    if sys.version_info < (3, 11):
        print(
            "Python 3.11 or higher is required. "
            "Please install Python 3.11 following the instructions in README.md."
        )
        return 1

    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        print("Warning: running under Rosetta; Metal acceleration unavailable.")
        if not args.allow_rosetta:
            print("Use '--allow-rosetta' to continue anyway.")
            return 1

    if not args.skip_deps:
        ensure_deps(install_optional=args.full_deps)

    from solhunter_zero import device

    device.ensure_gpu_env()
    import torch

    torch.set_default_device(device.get_default_device())
    config_path: str | None = None
    active_keypair: str | None = None
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    if not args.skip_setup:
        from solhunter_zero.config import load_config, validate_config

        ensure_config()
        config_path = os.getenv("SOLHUNTER_CONFIG")
        if not config_path:
            for name in ("config.yaml", "config.yml", "config.toml"):
                if Path(name).is_file():
                    config_path = name
                    break
        cfg = load_config(config_path)
        cfg = validate_config(cfg)
        if not args.skip_endpoint_check:
            ensure_endpoints(cfg)
        try:
            ensure_wallet_cli()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        ensure_default_keypair()
        ensure_keypair()
        from solhunter_zero import wallet

        active_keypair = wallet.get_active_keypair_name()

    if not args.skip_rpc_check:
        ensure_rpc(warn_only=args.one_click)
        rpc_status = "reachable"
    else:
        rpc_status = "skipped"

    gpu_device = str(device.get_default_device()) if device.detect_gpu() else "none"

    ensure_cargo()
    build_route_ffi()
    build_depth_service()

    if not args.skip_preflight:
        from scripts import preflight

        try:
            preflight.main()
        except SystemExit as exc:  # propagate non-zero exit codes
            code = exc.code if isinstance(exc.code, int) else 1
            if code:
                return code
    print("Startup summary:")
    print(f"  Config file: {config_path or 'none'}")
    print(f"  Active keypair: {active_keypair or 'none'}")
    print(f"  GPU device: {gpu_device}")
    print(f"  RPC endpoint: {rpc_url} ({rpc_status})")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "solhunter_zero.main", "--auto", *rest],
    )


def run(argv: list[str] | None = None) -> int:
    try:
        code = main(argv)
    except SystemExit as exc:
        code = exc.code
    except Exception:
        from scripts import diagnostics

        diagnostics.main()
        raise
    if code:
        from scripts import diagnostics

        diagnostics.main()
    return code or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
