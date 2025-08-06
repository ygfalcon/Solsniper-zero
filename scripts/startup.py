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
import contextlib
import io
from pathlib import Path
import json
from scripts import preflight
from solhunter_zero.bootstrap_utils import (
    _pip_install,
    ensure_deps,
    ensure_depth_service,
    ensure_route_ffi,
    ensure_venv,
)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from solhunter_zero import env  # noqa: E402

env.load_env_file(ROOT / ".env")
os.environ.setdefault("DEPTH_SERVICE", "true")
from solhunter_zero import device  # noqa: E402

if platform.system() == "Darwin" and platform.machine() == "x86_64":
    script = Path(__file__).resolve()
    cmd = ["arch", "-arm64", sys.executable, str(script), *sys.argv[1:]]
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:  # pragma: no cover - hard failure
        msg = (
            f"Failed to re-exec {script.name} via 'arch -arm64': {exc}\n"
            "Please use the unified entry point (scripts/launcher.py or start.command)."
        )
        raise SystemExit(msg)

MAX_PREFLIGHT_LOG_SIZE = 1_000_000  # 1 MB


def rotate_preflight_log(
    path: Path | None = None, max_bytes: int = MAX_PREFLIGHT_LOG_SIZE
) -> None:
    """Rotate or truncate the preflight log before writing new output.

    When ``path`` exists and exceeds ``max_bytes`` it is moved to ``.1``.
    Otherwise the file is truncated to start fresh for the current run.
    """

    path = path or ROOT / "preflight.log"
    if not path.exists():
        return
    try:
        if path.stat().st_size > max_bytes:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError:
        pass


def ensure_config() -> None:
    """Ensure a configuration file exists and is valid."""
    from solhunter_zero.config_bootstrap import ensure_config as _ensure_config

    _ensure_config()


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
        print(
            "Failed to install 'solhunter-wallet'. Please run 'pip install .' manually."
        )
        raise

    if shutil.which("solhunter-wallet") is None:
        print("'solhunter-wallet' still not available after installation. Aborting.")
        raise SystemExit(1)

def ensure_keypair() -> None:
    """Ensure a usable keypair exists and is selected."""

    import logging
    from pathlib import Path

    from solhunter_zero import wallet

    log = logging.getLogger(__name__)
    one_click = os.getenv("AUTO_SELECT_KEYPAIR") == "1"

    def _msg(msg: str) -> None:
        if one_click:
            log.info(msg)
        else:
            print(msg)

    keypair_json = os.environ.get("KEYPAIR_JSON")
    result = wallet.setup_default_keypair()
    name, mnemonic_path = result.name, result.mnemonic_path
    keypair_path = Path(wallet.KEYPAIR_DIR) / f"{name}.json"

    if keypair_json:
        _msg("Keypair saved from KEYPAIR_JSON and selected as 'default'.")
        _msg(f"Keypair stored at {keypair_path}.")
    elif mnemonic_path:
        _msg(f"Generated mnemonic and keypair '{name}'.")
        _msg(f"Keypair stored at {keypair_path}.")
        _msg(f"Mnemonic stored at {mnemonic_path}.")
        if not one_click:
            _msg("Please store this mnemonic securely; it will not be shown again.")
    else:
        _msg(f"Using keypair '{name}'.")

    return


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints.  This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable.  BirdEye is only checked when an API key is configured.
    """

    import urllib.error
    from solhunter_zero.http import check_endpoint

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
        try:
            check_endpoint(url)
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            print(
                f"Failed to reach {name} at {url} after 3 attempts: {exc}."
                " Check your network connection or configuration."
            )
            failed.append(name)

    if failed:
        raise SystemExit(1)


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
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

    def _run(cmd, *, desc: str, **kwargs) -> None:
        try:
            subprocess.check_call(cmd, stderr=subprocess.PIPE, text=True, **kwargs)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised indirectly
            with open(ROOT / "startup.log", "a", encoding="utf-8") as log:
                log.write(f"Command: {exc.cmd}\n")
                if exc.stderr:
                    log.write(f"stderr: {exc.stderr}\n")
            raise SystemExit(f"{desc} failed") from None

    if platform.system() == "Darwin":
        from scripts.mac_setup import apply_brew_env, ensure_tools

        ensure_tools()
        apply_brew_env()
        if shutil.which("brew") is None:
            print("Homebrew is required on macOS. See scripts/mac_setup.py")
            raise SystemExit(1)
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        print("Installing Rust toolchain via rustup...")
        _run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            desc="Rust toolchain installation",
            shell=True,
        )
        installed = True
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        _run(["cargo", "--version"], desc="Checking cargo version", stdout=subprocess.DEVNULL)
    except SystemExit:
        print("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise
    if installed and platform.system() == "Darwin" and platform.machine() == "arm64":
        _run(["rustup", "target", "add", "aarch64-apple-darwin"], desc="Adding aarch64-apple-darwin target")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - unusual failure
            with open(ROOT / "startup.log", "a", encoding="utf-8") as log:
                log.write(f"Command: {exc.cmd}\n")
                if exc.stderr:
                    log.write(f"stderr: {exc.stderr}\n")
            print("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode) from None
        if "aarch64-apple-darwin" not in targets:
            _run(["rustup", "target", "add", "aarch64-apple-darwin"], desc="Adding aarch64-apple-darwin target")

    missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
    if missing:
        if platform.system() == "Darwin" and shutil.which("brew") is not None:
            print(
                f"Missing {', '.join(missing)}. Attempting to install with Homebrew...",
            )
            _run(["brew", "install", "pkg-config", "cmake"], desc="Installing build tools with Homebrew")
            missing = [
                tool
                for tool in ("pkg-config", "cmake")
                if shutil.which(tool) is None
            ]
        if missing:
            names = " and ".join(missing)
            brew = " ".join(missing)
            print(
                f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. "
                f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script.",
            )
            raise SystemExit(1)

def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        os.environ["SOLHUNTER_SKIP_VENV"] = "1"
    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency check"
    )
    parser.add_argument(
        "--full-deps",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip config and wallet prompts"
    )
    parser.add_argument(
        "--skip-rpc-check",
        action="store_true",
        help="Skip Solana RPC availability check",
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
        "--self-test",
        action="store_true",
        help="Run bootstrap and preflight checks then exit",
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
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Suppress post-run diagnostics collection",
    )
    args, rest = parser.parse_known_args(argv)

    if args.self_test:
        from solhunter_zero.bootstrap import bootstrap
        from scripts import preflight
        import re

        b_code = 0
        try:
            bootstrap(one_click=True)
        except SystemExit as exc:
            b_code = exc.code if isinstance(exc.code, int) else 1
        except Exception:
            b_code = 1

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            try:
                preflight.main()
            except SystemExit as exc:
                p_code = exc.code if isinstance(exc.code, int) else 1
            else:
                p_code = 0
        out = stdout_buf.getvalue()
        sys.stdout.write(out)
        passes = len(re.findall(r": OK\b", out))
        fails = len(re.findall(r": FAIL\b", out))
        print(
            f"Self-test summary: bootstrap {'PASS' if b_code == 0 else 'FAIL'}, "
            f"preflight: {passes} passed, {fails} failed."
        )
        return b_code or p_code

    if args.diagnostics:
        from scripts import diagnostics

        diagnostics.main()
        return 0
    if args.one_click:
        rest = ["--non-interactive", *rest]

    if args.skip_deps:
        os.environ["SOLHUNTER_SKIP_DEPS"] = "1"
    if args.full_deps:
        os.environ["SOLHUNTER_INSTALL_OPTIONAL"] = "1"
    if args.skip_setup:
        os.environ["SOLHUNTER_SKIP_SETUP"] = "1"

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

    if not args.skip_preflight:
        rotate_preflight_log()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            try:
                preflight.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
            else:
                code = 0
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        sys.stdout.write(out)
        sys.stderr.write(err)
        try:
            with open(ROOT / "preflight.log", "a", encoding="utf-8") as log:
                log.write(out)
                log.write(err)
        except OSError:
            pass
        if code:
            return code

    from solhunter_zero.bootstrap import bootstrap

    bootstrap(one_click=args.one_click)

    from solhunter_zero import device
    import torch

    torch.set_default_device(device.get_default_device())
    gpu_available = device.detect_gpu()
    gpu_device = str(device.get_default_device()) if gpu_available else "none"
    if gpu_device == "none":
        print(
            "No GPU backend detected. Install a Metal-enabled PyTorch build or run "
            "scripts/mac_setup.py to enable GPU support."
        )
    os.environ["SOLHUNTER_GPU_AVAILABLE"] = "1" if gpu_available else "0"
    os.environ["SOLHUNTER_GPU_DEVICE"] = gpu_device
    config_path: str | None = None
    active_keypair: str | None = None
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    if not args.skip_setup:
        from solhunter_zero.config import (
            load_config,
            validate_config,
            find_config_file,
        )

        config_path = find_config_file()
        if config_path is None and args.one_click:
            try:
                from scripts import quick_setup

                quick_setup.main(["--auto", "--non-interactive"])
            except Exception:
                pass
            config_path = find_config_file()
        cfg = load_config(config_path)
        cfg = validate_config(cfg)
        if not args.skip_endpoint_check:
            ensure_endpoints(cfg)
        try:
            ensure_wallet_cli()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        from solhunter_zero import wallet

        active_keypair = wallet.get_active_keypair_name()

    if not args.skip_rpc_check:
        ensure_rpc(warn_only=args.one_click)
        rpc_status = "reachable"
    else:
        rpc_status = "skipped"

    ensure_cargo()
    print("Startup summary:")
    print(f"  Config file: {config_path or 'none'}")
    print(f"  Active keypair: {active_keypair or 'none'}")
    print(f"  GPU device: {gpu_device}")
    print(f"  RPC endpoint: {rpc_url} ({rpc_status})")

    proc = subprocess.run(
        [sys.executable, "-m", "solhunter_zero.main", "--auto", *rest]
    )

    if not args.no_diagnostics:
        from scripts import diagnostics

        info = diagnostics.collect()
        summary = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"Diagnostics summary: {summary}")
        out_path = Path("diagnostics.json")
        try:
            out_path.write_text(json.dumps(info, indent=2))
        except Exception:
            pass
        else:
            print(f"Full diagnostics written to {out_path}")

    return proc.returncode


def run(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--one-click" not in args_list:
        args_list.insert(0, "--one-click")
    try:
        code = main(args_list)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception:
        if "--no-diagnostics" not in args_list:
            from scripts import diagnostics

            diagnostics.main()
        raise
    if code and "--no-diagnostics" not in args_list:
        from scripts import diagnostics

        diagnostics.main()
    return code or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
