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
from concurrent.futures import ThreadPoolExecutor, as_completed

from solhunter_zero.paths import ROOT

from scripts import preflight  # noqa: E402
from scripts import deps  # noqa: E402
import solhunter_zero.bootstrap_utils as bootstrap_utils
from solhunter_zero import preflight_utils  # noqa: E402
from solhunter_zero.config import apply_env_overrides, load_config
from solhunter_zero.bootstrap_utils import (
    ensure_deps,
    ensure_venv,
    ensure_endpoints,
)
from solhunter_zero.rpc_utils import ensure_rpc

from solhunter_zero.logging_utils import (
    log_startup,
    setup_logging,
    rotate_preflight_log,
    STARTUP_LOG,
)  # noqa: E402

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.table import Table

console = Console()
def ensure_target(name: str) -> None:
    from solhunter_zero.build_utils import ensure_target as _ensure_target

    _ensure_target(name)

if platform.system() == "Darwin" and platform.machine() == "x86_64":
    script = Path(__file__).resolve()
    cmd = ["arch", "-arm64", sys.executable, str(script), *sys.argv[1:]]
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:  # pragma: no cover - hard failure
        msg = (
            f"Failed to re-exec {script.name} via 'arch -arm64': {exc}\n"
            "Please use 'python start.py'."
        )
        raise SystemExit(msg)

setup_logging("startup", path=STARTUP_LOG)
setup_logging("preflight")
from solhunter_zero import device  # noqa: E402


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available."""

    ensure_deps(ensure_wallet_cli=True)


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

def ensure_cargo() -> None:
    """Wrapper around :func:`bootstrap_utils.ensure_cargo` that syncs ROOT."""
    bootstrap_utils.ROOT = ROOT
    bootstrap_utils.ensure_cargo()


def _disk_space_required_bytes() -> int:
    """Return the minimum free bytes required based on configuration."""

    try:
        cfg = apply_env_overrides(load_config())
        limit_gb = float(cfg.get("offline_data_limit_gb", 50))
    except Exception:
        limit_gb = 50
    return int(limit_gb * (1024 ** 3))


def _main_impl(argv: list[str] | None = None) -> int:
    console.print(Panel.fit("[bold cyan]SolHunter Zero Startup[/]"), justify="center")
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
    parser.set_defaults(one_click=True)
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

    disk_required = _disk_space_required_bytes()

    # Track status for summary table
    disk_status = "unknown"
    internet_status = "skipped" if args.offline or args.skip_rpc_check else "unknown"
    config_status = "skipped" if args.skip_setup else "unknown"
    wallet_status = "skipped" if args.skip_setup else "unknown"

    # Run early environment checks before any heavy work
    with Progress(console=console, transient=True) as progress:
        disk_task = progress.add_task("Checking disk space...", total=1)
        ok, msg = preflight_utils.check_disk_space(disk_required)
        progress.advance(disk_task)
    disk_status = "passed" if ok else "failed"
    console.print(f"[green]{msg}[/]" if ok else f"[red]{msg}[/]")
    if not ok:
        log_startup("Disk space check failed")
        raise SystemExit(1)
    log_startup("Disk space check passed")

    if args.offline or args.skip_rpc_check:
        internet_status = "skipped"
        log_startup("Internet connectivity check skipped")
    else:
        print("Checking internet connectivity...")
        ok, msg = preflight_utils.check_internet()
        print(msg)
        if not ok:
            internet_status = "failed"
            log_startup("Internet connectivity check failed")
            raise SystemExit(1)
        internet_status = "passed"
        log_startup("Internet connectivity check passed")

    from solhunter_zero.config_bootstrap import ensure_config
    from solhunter_zero.config_utils import select_active_keypair
    from solhunter_zero import wallet

    cfg_data: dict = {}
    config_path: Path | None = None
    keypair_path: Path | None = None
    mnemonic_path: Path | None = None
    active_keypair: str | None = None
    ran_quick_setup = False

    if not args.skip_setup:
        from scripts.quick_setup import _is_placeholder

        def _has_placeholder(value: object) -> bool:
            if isinstance(value, str):
                return _is_placeholder(value)
            if isinstance(value, dict):
                return any(_has_placeholder(v) for v in value.values())
            if isinstance(value, list):
                return any(_has_placeholder(v) for v in value)
            return False

        with Progress(console=console, transient=True) as progress:
            cfg_task = progress.add_task("Ensuring configuration...", total=1)
            try:
                config_path, cfg_data = ensure_config()
            except (Exception, SystemExit):
                cfg_new = run_quick_setup()
                if not cfg_new:
                    console.print("[red]Failed to create configuration via quick setup[/]")
                    return 1
                config_path, cfg_data = ensure_config(cfg_new)
                ran_quick_setup = True
            progress.advance(cfg_task)

            if _has_placeholder(cfg_data):
                cfg_new = run_quick_setup()
                if not cfg_new:
                    console.print("[red]Failed to populate configuration via quick setup[/]")
                    return 1
                config_path, cfg_data = ensure_config(cfg_new)
                ran_quick_setup = True
                if _has_placeholder(cfg_data):
                    console.print("[red]Configuration still contains placeholder values[/]")
                    return 1

            config_status = str(config_path)

            wallet_task = progress.add_task("Ensuring wallet CLI...", total=1)
            try:
                ensure_wallet_cli()
            except SystemExit as exc:
                return exc.code if isinstance(exc.code, int) else 1
            progress.advance(wallet_task)

            key_task = progress.add_task("Selecting active keypair...", total=1)
            info = select_active_keypair(auto=True if ran_quick_setup else args.one_click)
            active_keypair = info.name
            keypair_path = Path(wallet.KEYPAIR_DIR) / f"{active_keypair}.json"
            mnemonic_path = info.mnemonic_path
            progress.advance(key_task)
            wallet_status = active_keypair
        console.print("[green]Configuration complete[/]")

    if args.offline:
        endpoint_status = "offline"
    elif args.skip_endpoint_check or args.skip_setup:
        endpoint_status = "skipped"
    else:
        with Progress(console=console, transient=True) as progress:
            ep_task = progress.add_task("Checking HTTP endpoints...", total=1)
            ensure_endpoints(cfg_data)
            progress.advance(ep_task)
        console.print("[green]HTTP endpoints reachable[/]")
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
        (ROOT / ".cache" / "deps-installed").unlink(missing_ok=True)
        device.MPS_SENTINEL.unlink(missing_ok=True)

    if args.self_test:
        from solhunter_zero.bootstrap import bootstrap
        import re

        ok, msg = preflight_utils.check_disk_space(disk_required)
        if not ok:
            print(msg)
            raise SystemExit(1)
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
        with Progress(console=console, transient=True) as progress:
            with ThreadPoolExecutor() as executor:
                task_map = {
                    executor.submit(
                        ensure_deps, install_optional=args.full_deps
                    ): progress.add_task("Installing dependencies...", total=1),
                    executor.submit(ensure_target, "protos"): progress.add_task(
                        "Generating protos...", total=1
                    ),
                    executor.submit(ensure_target, "route_ffi"): progress.add_task(
                        "Building route FFI...", total=1
                    ),
                    executor.submit(ensure_target, "depth_service"): progress.add_task(
                        "Building depth service...", total=1
                    ),
                }
                for future in as_completed(task_map):
                    task_id = task_map[future]
                    task_desc = progress.tasks[task_id].description
                    try:
                        future.result()
                        progress.advance(task_id)
                    except Exception as exc:  # pragma: no cover - defensive
                        progress.update(task_id, description=f"{task_desc} [failed]", advance=1)
                        console.print(f"[red]{task_desc} failed: {exc}[/]")
                        return 1
        console.print("[green]Dependencies installed[/]")
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

    if args.skip_preflight:
        os.environ["SOLHUNTER_SKIP_PREFLIGHT"] = "1"
    else:
        results = preflight.run_preflight()
        failures: list[tuple[str, str]] = []
        for name, ok, msg in results:
            status = "OK" if ok else "FAIL"
            line = f"{name}: {status} - {msg}"
            sys.stdout.write(line + "\n")
            if not ok:
                failures.append((name, msg))
        if failures:
            summary = "; ".join(f"{n}: {m}" for n, m in failures)
            print(f"Preflight checks failed: {summary}")
            log_startup(f"Preflight checks failed: {summary}")
            sys.exit(1)

    if args.offline:
        rpc_status = "offline"
    elif args.skip_rpc_check:
        rpc_status = "skipped"
    else:
        ensure_rpc(warn_only=args.one_click)
        rpc_status = "reachable"
    from solhunter_zero.bootstrap import bootstrap

    # ``bootstrap`` performs its own config and keypair setup.  These steps have
    # already been handled above, so instruct it to skip them to avoid duplicate
    # work and simplify testing.
    os.environ["SOLHUNTER_SKIP_SETUP"] = "1"
    if args.no_diagnostics:
        os.environ["SOLHUNTER_NO_DIAGNOSTICS"] = "1"

    try:
        bootstrap(one_click=args.one_click)
    finally:
        os.environ.pop("SOLHUNTER_SKIP_SETUP", None)

    gpu_env = device.initialize_gpu()
    gpu_device = gpu_env.get("SOLHUNTER_GPU_DEVICE", "unknown")
    rpc_url = os.environ.get(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )

    log_startup(f"GPU device: {gpu_device}")
    log_startup(f"RPC endpoint: {rpc_url} ({rpc_status})")

    os.environ.pop("SOLHUNTER_SKIP_DEPS", None)

    ensure_cargo()
    log_startup_info(
        config_path=config_path,
        keypair_path=keypair_path,
        mnemonic_path=mnemonic_path,
        active_keypair=active_keypair,
    )
    summary_rows = [
        ("Disk space", disk_status),
        ("Internet", internet_status),
        ("Configuration", str(config_status)),
        ("Wallet", str(wallet_status)),
        ("HTTP endpoints", endpoint_status),
    ]

    table = Table(title="Startup Summary")
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    for item, status in summary_rows:
        table.add_row(item, status)
        log_startup(f"{item}: {status}")
    console.print(table)

    from solhunter_zero.agent_manager import AgentManager

    try:
        if AgentManager.from_config(load_config("config.toml")) is None:
            log_startup("AgentManager.from_config returned None")
            print("AgentManager.from_config returned None")
            return 1
    except Exception as exc:
        log_startup(f"AgentManager initialization failed: {exc}")
        print(f"Failed to initialize AgentManager: {exc}")
        return 1

    proc = subprocess.run(
        [sys.executable, "scripts/start_all.py", *rest],
        capture_output=True,
        text=True,
    )

    if proc.returncode == 0:
        msg = "SolHunter Zero launch complete – system ready."
        print(msg)
        log_startup(msg)
    else:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        for line in (proc.stdout + proc.stderr).splitlines():
            if line:
                log_startup(line)
        msg = f"SolHunter Zero startup failed with exit code {proc.returncode}"
        print(msg)
        log_startup(msg)

    # Diagnostics are handled by :func:`solhunter_zero.bootstrap.bootstrap`.

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

    log_path = STARTUP_LOG
    print("Log summary:")
    print(f"  Detailed logs: {log_path}")
    log_startup(f"Log summary: see {log_path}")

    return proc.returncode or hc_code


def main(argv: list[str] | None = None) -> int:
    prev_skip_venv = os.environ.get("SOLHUNTER_SKIP_VENV")
    if argv is not None:
        os.environ["SOLHUNTER_SKIP_VENV"] = "1"
    try:
        return _main_impl(argv)
    finally:
        if prev_skip_venv is None:
            os.environ.pop("SOLHUNTER_SKIP_VENV", None)
        else:
            os.environ["SOLHUNTER_SKIP_VENV"] = prev_skip_venv


def run(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
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
