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
        _pip_install(".[uvloop]")

    if "torch" in opt and platform.system() == "Darwin" and platform.machine() == "arm64":
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

    if platform.system() == "Darwin" and platform.machine() == "arm64":
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
            _pip_install(f".[{','.join(extras)}]")
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            _pip_install(pkg)

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
        _pip_install(".")
    except SystemExit:
        print("Failed to install 'solhunter-wallet'. Please run 'pip install .' manually.")
        raise

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

def ensure_cargo() -> None:
    installed = False
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        if platform.system() == "Darwin":
            if shutil.which("brew") is None:
                print(
                    "Homebrew is required to install the Rust toolchain. "
                    "Install it by running scripts/mac_setup.sh and re-run this script.",
                )
                raise SystemExit(1)
            try:
                subprocess.check_call(
                    ["xcode-select", "-p"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                print(
                    "Xcode command line tools are required. Launching installer..."
                )
                try:
                    subprocess.check_call(["xcode-select", "--install"])
                except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
                    print(
                        f"Failed to start Xcode command line tools installer: {exc}"
                    )
                else:
                    print(
                        "The installer may prompt for confirmation; "
                        "after it finishes, re-run this script to resume."
                    )
                raise SystemExit(1)
        print("Installing Rust toolchain via rustup...")
        subprocess.check_call(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            shell=True,
        )
        installed = True
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        subprocess.check_call(["cargo", "--version"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        print("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise SystemExit(exc.returncode)
    if (
        installed
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
    ):
        subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:
            print("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
    if missing:
        names = " and ".join(missing)
        brew = " ".join(missing)
        print(
            f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. "
            f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script."
        )
        raise SystemExit(1)


def ensure_route_ffi() -> None:
    """Ensure the ``route_ffi`` Rust library is built and copied locally.

    The Python package expects ``solhunter_zero/libroute_ffi.{so|dylib}`` to be
    present.  When missing, this function invokes ``cargo build`` for the
    ``route_ffi`` crate and copies the resulting shared library into the
    package directory.  On Apple Silicon the build target is explicitly set to
    ``aarch64-apple-darwin`` to match the host architecture.
    """

    libname = "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    libpath = ROOT / "solhunter_zero" / libname
    if libpath.exists():
        return

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            print("Failed to verify rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in installed_targets:
            subprocess.check_call(
                ["rustup", "target", "add", "aarch64-apple-darwin"]
            )

    cmd = [
        "cargo",
        "build",
        "--manifest-path",
        str(ROOT / "route_ffi" / "Cargo.toml"),
        "--release",
    ]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        cmd.extend(["--target", "aarch64-apple-darwin"])

    subprocess.check_call(cmd)

    target_dir = ROOT / "route_ffi" / "target"
    candidates = [target_dir / "release" / libname]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        candidates.append(target_dir / "aarch64-apple-darwin" / "release" / libname)

    for built in candidates:
        if built.exists():
            shutil.copy2(built, libpath)
            break

    if not libpath.exists():
        print(f"Warning: failed to locate built {libname}; please build manually.")


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
        rest = ["--non-interactive", *rest]
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")
        preflight = subprocess.run(
            [sys.executable, "scripts/preflight.py"],
            capture_output=True,
            text=True,
        )
        if preflight.returncode != 0:
            if preflight.stdout:
                print(preflight.stdout, end="")
            if preflight.stderr:
                print(preflight.stderr, end="", file=sys.stderr)
            return preflight.returncode

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

    import torch
    from solhunter_zero import device

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
    ensure_route_ffi()
    print("Startup summary:")
    print(f"  Config file: {config_path or 'none'}")
    print(f"  Active keypair: {active_keypair or 'none'}")
    print(f"  GPU device: {gpu_device}")
    print(f"  RPC endpoint: {rpc_url} ({rpc_status})")
    run_sh = ROOT / "run.sh"
    if os.name != "nt" and run_sh.is_file() and os.access(run_sh, os.X_OK):
        os.execv(str(run_sh), [str(run_sh), "--auto", *rest])
    else:
        os.execv(sys.executable, [sys.executable, "-m", "solhunter_zero.main", "--auto", *rest])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
