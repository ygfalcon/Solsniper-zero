#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""

from __future__ import annotations

import sys

import argparse
import os
import platform
import subprocess
import shutil
import contextlib
import io
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

MAX_STARTUP_LOG_SIZE = 1_000_000  # 1 MB


def rotate_startup_log(path: Path = ROOT / "startup.log") -> None:
    """Rotate or truncate the startup log before writing new output."""

    if not path.exists():
        return
    try:
        if path.stat().st_size > MAX_STARTUP_LOG_SIZE:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError:
        pass

rotate_startup_log()

from scripts import preflight  # noqa: E402
from scripts import deps  # noqa: E402
import solhunter_zero.bootstrap_utils as bootstrap_utils
from solhunter_zero.bootstrap_utils import (
    _pip_install,
    ensure_deps,
    ensure_depth_service,
    ensure_route_ffi,
    ensure_venv,
)

from solhunter_zero import env  # noqa: E402
from solhunter_zero.logging_utils import log_startup  # noqa: E402

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


def ensure_config() -> Path:
    """Ensure a configuration file exists and is valid and return its path."""
    from solhunter_zero.config_bootstrap import ensure_config as _ensure_config

    return _ensure_config()


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

def ensure_keypair() -> tuple["wallet.KeypairInfo", Path]:
    """Ensure a usable keypair exists and is selected.

    Returns the :class:`~solhunter_zero.wallet.KeypairInfo` and path to the
    JSON keypair file.
    """

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
    try:
        result = wallet.setup_default_keypair()
    except Exception as exc:  # pragma: no cover - handled interactively
        print(f"Failed to set up default keypair: {exc}")
        if keypair_json:
            os.environ.pop("KEYPAIR_JSON", None)
            print("Removed KEYPAIR_JSON environment variable.")
        if one_click:
            raise SystemExit(1)
        input(
            "Press Enter to retry without KEYPAIR_JSON or Ctrl+C to abort..."
        )
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

    return result, keypair_path


def log_startup_info(*, config_path: Path | None = None, keypair_path: Path | None = None,
                     mnemonic_path: Path | None = None, active_keypair: str | None = None) -> None:
    """Append startup details to ``startup.log``."""

    lines: list[str] = []
    if config_path:
        lines.append(f"Config path: {config_path}")
    if keypair_path:
        lines.append(f"Keypair path: {keypair_path}")
    if mnemonic_path:
        lines.append(f"Mnemonic path: {mnemonic_path}")
    if active_keypair:
        lines.append(f"Active keypair: {active_keypair}")
    if not lines:
        return
    for line in lines:
        log_startup(line)


def run_quick_setup() -> str | None:
    """Run the quick setup non-interactively and return new config path."""
    try:
        from scripts import quick_setup
        from solhunter_zero.config import find_config_file

        quick_setup.main(["--auto", "--non-interactive"])
        return find_config_file()
    except Exception:
        return None


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints.  This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable.  BirdEye is only checked when an API key is configured.
    """

    import asyncio
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

    async def _check(name: str, url: str) -> tuple[str, Exception] | None:
        # Each URL is checked with its own exponential backoff.
        for attempt in range(3):
            try:
                # ``check_endpoint`` is synchronous; run it in a thread to avoid blocking.
                await asyncio.to_thread(check_endpoint, url, retries=1)
                return None
            except urllib.error.URLError as exc:  # pragma: no cover - network failure
                if attempt == 2:
                    return name, exc
                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1} failed for {name} at {url}: {exc}. "
                    f"Retrying in {wait} seconds..."
                )
                await asyncio.sleep(wait)

    async def _run() -> list[tuple[str, Exception] | None]:
        tasks = [_check(name, url) for name, url in urls.items()]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    failures: list[tuple[str, str, Exception]] = []
    for name_exc in results:
        if name_exc is None:
            continue
        name, exc = name_exc
        url = urls[name]
        failures.append((name, url, exc))

    if failures:
        details = "; ".join(
            f"{name} at {url} ({exc})" for name, url, exc in failures
        )
        print(
            "Failed to reach the following endpoints: "
            f"{details}. Check your network connection or configuration."
        )
        raise SystemExit(1)


def check_disk_space(min_bytes: int) -> None:
    """Ensure there is at least ``min_bytes`` free on the current filesystem.

    The check uses the repository root path to determine available free space.
    If the requirement is not met, an instructive message is printed and the
    process exits.
    """

    try:
        _, _, free = shutil.disk_usage(ROOT)
    except OSError as exc:  # pragma: no cover - unexpected failure
        print(f"Unable to determine free disk space: {exc}")
        raise SystemExit(1)

    if free < min_bytes:
        required_gb = min_bytes / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        print(
            f"Insufficient disk space: {free_gb:.2f} GB available,"
            f" {required_gb:.2f} GB required."
        )
        print("Please free up disk space and try again.")
        raise SystemExit(1)


def check_internet(url: str = "https://example.com") -> None:
    """Ensure basic internet connectivity by reaching a known host.

    The function performs a simple ``GET`` request to ``url`` with
    exponential backoff.  If all attempts fail, startup is aborted with a
    clear error message.
    """

    import urllib.request
    import urllib.error
    import time

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                resp.read()
                return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                print(
                    f"Failed to reach {url} after 3 attempts: {exc}. "
                    "Check your internet connection."
                )
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to reach {url}: {exc}. "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)


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
    """Wrapper around :func:`bootstrap_utils.ensure_cargo` that syncs ROOT."""
    bootstrap_utils.ROOT = ROOT
    bootstrap_utils.ensure_cargo()


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
        "--offline",
        action="store_true",
        help="Skip internet and RPC connectivity checks",
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
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Force macOS setup and clear dependency caches",
    )
    args, rest = parser.parse_known_args(argv)

    if args.repair and platform.system() == "Darwin":
        from scripts import mac_setup

        report = mac_setup.prepare_macos_env(non_interactive=True)
        mac_setup.apply_brew_env()
        for step, info in report.get("steps", {}).items():
            msg = info.get("message", "")
            if msg:
                print(f"{step}: {info['status']} - {msg}")
            else:
                print(f"{step}: {info['status']}")
            if info.get("status") == "error":
                fix = mac_setup.MANUAL_FIXES.get(step)
                if fix:
                    print(f"Manual fix for {step}: {fix}")
        # Clear cache markers so subsequent steps rerun fully
        (ROOT / ".cache" / "cargo-installed").unlink(missing_ok=True)
        device.MPS_SENTINEL.unlink(missing_ok=True)

    if args.self_test:
        from solhunter_zero.bootstrap import bootstrap
        from scripts import preflight
        import re

        check_disk_space(1 << 30)
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
    if args.skip_setup or args.one_click:
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

    if args.offline:
        rpc_status = "offline"
    elif args.skip_rpc_check:
        rpc_status = "skipped"
    else:
        check_internet()
        ensure_rpc(warn_only=args.one_click)
        rpc_status = "reachable"

    check_disk_space(1 << 30)
    from solhunter_zero.bootstrap import bootstrap

    bootstrap(one_click=args.one_click)

    config_path: Path | None = None
    keypair_path: Path | None = None
    mnemonic_path: Path | None = None
    active_keypair: str | None = None

    if args.one_click and not args.skip_setup:
        cfg_path = ensure_config()
        kp_info, keypair_path = ensure_keypair()
        config_path = cfg_path
        mnemonic_path = kp_info.mnemonic_path
        active_keypair = kp_info.name

    gpu_env = device.ensure_gpu_env()
    gpu_available = os.environ.get("SOLHUNTER_GPU_AVAILABLE") == "1"
    gpu_device = os.environ.get("SOLHUNTER_GPU_DEVICE", "none")
    if gpu_env:
        print(
            "Configured GPU environment: "
            + ", ".join(f"{k}={v}" for k, v in gpu_env.items())
        )
    if not gpu_available:
        print(
            "No GPU backend detected. Install a Metal-enabled PyTorch build or run "
            "scripts/mac_setup.py to enable GPU support."
        )
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    if not args.skip_setup:
        from solhunter_zero.config import (
            load_config,
            validate_config,
            find_config_file,
        )

        if config_path is None:
            cfg_path = find_config_file()
            if cfg_path is None:
                cfg_path = run_quick_setup()
            if cfg_path is not None:
                config_path = Path(cfg_path)
        if config_path is not None:
            cfg = load_config(config_path)
            cfg = validate_config(cfg)
        else:
            cfg = {}
        if not args.skip_endpoint_check:
            ensure_endpoints(cfg)
        try:
            ensure_wallet_cli()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        from solhunter_zero import wallet
        if active_keypair is None:
            active_keypair = wallet.get_active_keypair_name()
        if keypair_path is None and active_keypair:
            keypair_path = Path(wallet.KEYPAIR_DIR) / f"{active_keypair}.json"

    ensure_cargo()
    log_startup_info(
        config_path=config_path,
        keypair_path=keypair_path,
        mnemonic_path=mnemonic_path,
        active_keypair=active_keypair,
    )
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
