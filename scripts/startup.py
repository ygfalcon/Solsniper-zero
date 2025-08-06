#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""

from __future__ import annotations

import sys

import argparse
import os
import platform
import subprocess
import shutil
from pathlib import Path

from scripts import deps

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEPTH_SERVICE", "true")

if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_venv(argv: list[str] | None) -> None:
    """Create a local virtual environment and re-invoke the script inside it.

    The environment is created in ``ROOT/.venv`` using ``python -m venv``.  If
    the script is not already executing from that interpreter, ``execv`` is used
    to restart the current process with ``.venv/bin/python`` (or the Windows
    equivalent).  When ``argv`` is not ``None`` the function assumes it is being
    called from tests and does nothing to avoid spawning subprocesses.
    """

    if argv is not None:  # avoid side effects during tests
        return

    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        print("Creating virtual environment in .venv...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    if Path(sys.prefix) != venv_dir:
        python = (
            venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        )
        os.execv(str(python), [str(python), *sys.argv])


def ensure_deps() -> None:
    req, opt = deps.check_deps()
    if not req and not opt:
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
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", ".[uvloop]"]
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
            print(f"Failed to install required dependencies: {exc}")
            raise SystemExit(exc.returncode)

    if "torch" in opt and platform.system() == "Darwin" and platform.machine() == "arm64":
        print(
            "Installing torch==2.1.0 and torchvision==0.16.0 for macOS arm64 with Metal support..."
        )
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    # Pinned versions: update together when upgrading Metal wheels.
                    "torch==2.1.0",
                    "torchvision==0.16.0",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/metal",
                ]
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
            print(f"Failed to install torch with Metal wheels: {exc}")
            raise SystemExit(exc.returncode)
        else:
            opt.remove("torch")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import importlib
        import torch
        if not torch.backends.mps.is_available():
            print(
                "MPS backend not available; attempting to reinstall Metal wheel..."
            )
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--force-reinstall",
                        "torch==2.1.0",
                        "torchvision==0.16.0",
                        "--extra-index-url",
                        "https://download.pytorch.org/whl/metal",
                    ]
                )
            except subprocess.CalledProcessError as exc:
                print(f"Failed to reinstall torch with Metal wheels: {exc}")
            importlib.reload(torch)
            if not torch.backends.mps.is_available():
                print(
                    "WARNING: MPS backend still not available; continuing with CPU."
                )

    if opt:
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
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f".[{','.join(extras)}]",
                    ]
                )
            except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
                print(f"Failed to install optional extras '{extras}': {exc}")
                raise SystemExit(exc.returncode)
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg]
                )
            except subprocess.CalledProcessError as exc:  # pragma: no cover - network failure
                print(f"Failed to install optional dependency '{pkg}': {exc}")
                raise SystemExit(exc.returncode)

    req_after, opt_after = deps.check_deps()
    if req_after or opt_after:
        if req_after:
            print(
                "Missing required dependencies after installation: "
                + ", ".join(req_after)
            )
        if opt_after:
            print(
                "Missing optional dependencies after installation: "
                + ", ".join(opt_after)
            )
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
        print(
            "Failed to install 'solhunter-wallet'. Please run 'pip install .' manually."
        )
        raise SystemExit(exc.returncode)

    if shutil.which("solhunter-wallet") is None:
        print("'solhunter-wallet' still not available after installation. Aborting.")
        raise SystemExit(1)


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


def ensure_rpc() -> None:
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
                print(
                    f"Failed to contact Solana RPC at {rpc_url} after 3 attempts: {exc}."
                    " Please ensure the endpoint is reachable or set SOLANA_RPC_URL to a valid RPC."
                )
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}."
                f" Retrying in {wait} seconds..."
            )
            time.sleep(wait)


def ensure_cargo() -> None:
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        if platform.system() == "Darwin":
            try:
                subprocess.check_call(
                    ["xcode-select", "-p"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                print(
                    "Xcode command line tools are required. Install them with 'xcode-select --install'.",
                )
                raise SystemExit(1)
        print("Installing Rust toolchain via rustup...")
        subprocess.check_call(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            shell=True,
        )
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        subprocess.check_call(["cargo", "--version"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        print("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise SystemExit(exc.returncode)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:
            print("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])


def main(argv: list[str] | None = None) -> int:
    ensure_venv(argv)

    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
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
        "--one-click",
        action="store_true",
        help="Enable fully automated non-interactive startup",
    )
    parser.add_argument(
        "--allow-rosetta",
        action="store_true",
        help="Allow running under Rosetta (no Metal acceleration)",
    )
    args, rest = parser.parse_known_args(argv)

    if args.one_click:
        args.skip_rpc_check = True
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
        ensure_deps()
    if not args.skip_setup:
        from solhunter_zero.config import load_config, validate_config

        ensure_config()
        cfg = load_config()
        cfg = validate_config(cfg)
        if not args.skip_endpoint_check:
            ensure_endpoints(cfg)
        ensure_keypair()

    if not args.skip_rpc_check:
        ensure_rpc()
    ensure_cargo()
    run_sh = ROOT / "run.sh"
    if os.name != "nt" and run_sh.is_file() and os.access(run_sh, os.X_OK):
        os.execv(str(run_sh), [str(run_sh), "--auto", *rest])
    else:
        os.execv(sys.executable, [sys.executable, "-m", "solhunter_zero.main", "--auto", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
