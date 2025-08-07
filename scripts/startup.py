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

from scripts import preflight  # noqa: E402
from scripts import deps  # noqa: E402
import solhunter_zero.bootstrap_utils as bootstrap_utils
from solhunter_zero.bootstrap_utils import (
    ensure_deps,
    ensure_venv,
    ensure_endpoints,
)

import solhunter_zero.env_config as env_config  # noqa: E402
from solhunter_zero.logging_utils import log_startup, rotate_startup_log  # noqa: E402

rotate_startup_log()
env_config.configure_environment(ROOT)
from solhunter_zero import device  # noqa: E402


def ensure_route_ffi() -> None:
    from solhunter_zero.build_utils import ensure_route_ffi as _ensure_route_ffi

    _ensure_route_ffi()


def ensure_depth_service() -> None:
    from solhunter_zero.build_utils import ensure_depth_service as _ensure_depth_service

    _ensure_depth_service()

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


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available."""

    if shutil.which("solhunter-wallet") is None:
        print(
            "'solhunter-wallet' command not found. Install the package providing it and re-run.",
        )
        raise SystemExit(1)


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

    from solhunter_zero.config_utils import ensure_default_config, select_active_keypair
    from solhunter_zero.config import load_config, validate_config
    from solhunter_zero import wallet

    cfg_data: dict = {}
    config_path: Path | None = None
    keypair_path: Path | None = None
    mnemonic_path: Path | None = None
    active_keypair: str | None = None

    if not args.skip_setup:
        try:
            config_path = ensure_default_config()
            cfg_data = validate_config(load_config(config_path))
        except (Exception, SystemExit) as exc:
            print(f"Invalid configuration: {exc}")
            try:
                resp = input(
                    "Run quick setup to regenerate configuration? [y/N]: "
                )
            except EOFError:
                resp = ""
            if resp.strip().lower() in {"y", "yes"}:
                from scripts.quick_setup import run as quick_setup_run

                quick_setup_run()
            return 1
        try:
            ensure_wallet_cli()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        info = select_active_keypair(auto=args.one_click)
        active_keypair = info.name
        keypair_path = Path(wallet.KEYPAIR_DIR) / f"{active_keypair}.json"
        mnemonic_path = info.mnemonic_path

    if args.offline:
        endpoint_status = "offline"
    elif args.skip_endpoint_check or args.skip_setup:
        endpoint_status = "skipped"
    else:
        ensure_endpoints(cfg_data)
        endpoint_status = "reachable"

    if args.repair and platform.system() == "Darwin":
        from solhunter_zero import macos_setup

        report = macos_setup.prepare_macos_env(non_interactive=True)
        for step, info in report.get("steps", {}).items():
            msg = info.get("message", "")
            if msg:
                print(f"{step}: {info['status']} - {msg}")
            else:
                print(f"{step}: {info['status']}")
            if info.get("status") == "error":
                fix = macos_setup.MANUAL_FIXES.get(step)
                if fix:
                    print(f"Manual fix for {step}: {fix}")
        # Clear cache markers so subsequent steps rerun fully
        (ROOT / ".cache" / "cargo-installed").unlink(missing_ok=True)
        device.MPS_SENTINEL.unlink(missing_ok=True)

    if args.self_test:
        from solhunter_zero.bootstrap import bootstrap
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

    if not args.skip_deps:
        ensure_deps(install_optional=args.full_deps)
        ensure_route_ffi()
        ensure_depth_service()
    os.environ["SOLHUNTER_SKIP_DEPS"] = "1"
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
        results = preflight.run_preflight()
        log_lines: list[str] = []
        failures: list[tuple[str, str]] = []
        for name, ok, msg in results:
            status = "OK" if ok else "FAIL"
            line = f"{name}: {status} - {msg}\n"
            sys.stdout.write(line)
            log_lines.append(line)
            if not ok:
                failures.append((name, msg))
        try:
            with open(ROOT / "preflight.log", "a", encoding="utf-8") as log:
                log.writelines(log_lines)
        except OSError:
            pass
        for name, msg in failures:
            log_startup(f"Preflight failure: {name} - {msg}")
        if failures:
            return 1

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

    # ``bootstrap`` performs its own config and keypair setup.  These steps have
    # already been handled above, so instruct it to skip them to avoid duplicate
    # work and simplify testing.
    os.environ["SOLHUNTER_SKIP_SETUP"] = "1"
    try:
        bootstrap(one_click=args.one_click)
    finally:
        os.environ.pop("SOLHUNTER_SKIP_SETUP", None)

    gpu_env = device.initialize_gpu()
    gpu_device = gpu_env.get("SOLHUNTER_GPU_DEVICE", "unknown")
    rpc_url = os.environ.get(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )

    os.environ.pop("SOLHUNTER_SKIP_DEPS", None)

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
    print(f"  HTTP endpoints: {endpoint_status}")

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

    # Run a post-execution health check and append the results to startup.log.
    from scripts import healthcheck

    selected = list(preflight.CHECKS)
    critical = {name for name, _ in selected}
    non_critical = {"Homebrew", "Rustup", "Rust", "Xcode CLT", "GPU"}
    critical -= non_critical
    if args.skip_deps:
        selected = [c for c in selected if c[0] != "Dependencies"]
        critical.discard("Dependencies")
    if args.skip_setup:
        selected = [c for c in selected if c[0] not in {"Config", "Keypair"}]
        critical.difference_update({"Config", "Keypair"})
    if args.skip_rpc_check or args.offline:
        selected = [c for c in selected if c[0] != "Network"]
        critical.discard("Network")
    if args.skip_preflight:
        selected = []
        critical = set()

    hc_out = io.StringIO()
    hc_err = io.StringIO()
    with contextlib.redirect_stdout(hc_out), contextlib.redirect_stderr(hc_err):
        try:
            hc_code = healthcheck.main(selected, critical=critical)
        except SystemExit as exc:  # pragma: no cover - defensive
            hc_code = exc.code if isinstance(exc.code, int) else 1

    out = hc_out.getvalue()
    err = hc_err.getvalue()
    sys.stdout.write(out)
    sys.stderr.write(err)
    for line in (out + err).splitlines():
        if line:
            log_startup(line)

    return proc.returncode or hc_code


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
