from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import subprocess
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.progress import Progress

from solhunter_zero.paths import ROOT
from scripts import preflight  # noqa: E402
from scripts import deps  # noqa: E402
import solhunter_zero.bootstrap_utils as bootstrap_utils
from solhunter_zero import preflight_utils  # noqa: E402
from solhunter_zero import device  # noqa: E402

console = Console()


def ensure_target(name: str) -> None:
    from solhunter_zero.build_utils import ensure_target as _ensure_target

    _ensure_target(name)


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available."""
    ensure_deps(ensure_wallet_cli=True)  # type: ignore[name-defined]


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


def _disk_space_required_bytes(apply_env_overrides, load_config) -> int:
    try:
        cfg = apply_env_overrides(load_config())
        limit_gb = float(cfg.get("offline_data_limit_gb", 50))
    except Exception:
        limit_gb = 50
    return int(limit_gb * (1024 ** 3))


@contextlib.contextmanager
def temporary_env(key: str, value: str):
    """Temporarily set an environment variable."""
    original = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def check_disk_space(disk_required: int, log_startup) -> str:
    """Check available disk space."""
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Checking disk space...", total=1)
        ok, msg = preflight_utils.check_disk_space(disk_required)
        progress.advance(task)
    console.print(f"[green]{msg}[/]" if ok else f"[red]{msg}[/]")
    if not ok:
        log_startup("Disk space check failed")
        raise SystemExit(1)
    log_startup("Disk space check passed")
    return "passed"


def check_network(args, log_startup) -> str:
    """Check internet connectivity unless skipped."""
    if args.offline or args.skip_rpc_check:
        log_startup("Internet connectivity check skipped")
        return "skipped"
    console.print("Checking internet connectivity...")
    ok, msg = preflight_utils.check_internet()
    console.print(msg)
    if not ok:
        log_startup("Internet connectivity check failed")
        raise SystemExit(1)
    log_startup("Internet connectivity check passed")
    return "passed"


def ensure_configuration_and_wallet(args, ensure_wallet_cli, run_quick_setup):
    """Ensure configuration file and wallet CLI exist and select a keypair."""
    if args.skip_setup:
        return None, {}, None, None, None, "skipped", "skipped"
    from scripts.quick_setup import _is_placeholder
    from solhunter_zero.config_bootstrap import ensure_config
    from solhunter_zero.config_utils import select_active_keypair
    from solhunter_zero import wallet

    def _has_placeholder(value: object) -> bool:
        if isinstance(value, str):
            return _is_placeholder(value)
        if isinstance(value, dict):
            return any(_has_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(_has_placeholder(v) for v in value)
        return False

    cfg_data: dict = {}
    config_path: Path | None = None
    keypair_path: Path | None = None
    mnemonic_path: Path | None = None
    active_keypair: str | None = None
    ran_quick_setup = False
    with Progress(console=console, transient=True) as progress:
        cfg_task = progress.add_task("Ensuring configuration...", total=1)
        try:
            config_path, cfg_data = ensure_config()
        except (Exception, SystemExit):
            cfg_new = run_quick_setup()
            if not cfg_new:
                console.print("[red]Failed to create configuration via quick setup[/]")
                raise SystemExit(1)
            config_path, cfg_data = ensure_config(cfg_new)
            ran_quick_setup = True
        progress.advance(cfg_task)

        if _has_placeholder(cfg_data):
            cfg_new = run_quick_setup()
            if not cfg_new:
                console.print("[red]Failed to populate configuration via quick setup[/]")
                raise SystemExit(1)
            config_path, cfg_data = ensure_config(cfg_new)
            ran_quick_setup = True
            if _has_placeholder(cfg_data):
                console.print("[red]Configuration still contains placeholder values[/]")
                raise SystemExit(1)

        wallet_task = progress.add_task("Ensuring wallet CLI...", total=1)
        ensure_wallet_cli()
        progress.advance(wallet_task)

        key_task = progress.add_task("Selecting active keypair...", total=1)
        info = select_active_keypair(auto=True if ran_quick_setup else args.one_click)
        active_keypair = info.name
        keypair_path = Path(wallet.KEYPAIR_DIR) / f"{active_keypair}.json"
        mnemonic_path = info.mnemonic_path
        progress.advance(key_task)
    console.print("[green]Configuration complete[/]")
    return (
        config_path,
        cfg_data,
        keypair_path,
        mnemonic_path,
        active_keypair,
        str(config_path),
        active_keypair,
    )


def check_endpoints(args, cfg_data, ensure_endpoints) -> str:
    """Verify HTTP endpoints unless skipped."""
    if args.offline:
        return "offline"
    if args.skip_endpoint_check or args.skip_setup:
        return "skipped"
    with Progress(console=console, transient=True) as progress:
        ep_task = progress.add_task("Checking HTTP endpoints...", total=1)
        ensure_endpoints(cfg_data)
        progress.advance(ep_task)
    console.print("[green]HTTP endpoints reachable[/]")
    return "reachable"


def install_dependencies(args, ensure_deps, ensure_target) -> None:
    """Install required dependencies unless skipped."""
    if args.skip_deps:
        return
    with Progress(console=console, transient=True) as progress:
        with ThreadPoolExecutor() as executor:
            task_map = {
                executor.submit(ensure_deps, install_optional=args.full_deps): progress.add_task("Installing dependencies...", total=1),
                executor.submit(ensure_target, "protos"): progress.add_task("Generating protos...", total=1),
                executor.submit(ensure_target, "route_ffi"): progress.add_task("Building route FFI...", total=1),
                executor.submit(ensure_target, "depth_service"): progress.add_task("Building depth service...", total=1),
            }
            for future in as_completed(task_map):
                task_id = task_map[future]
                task_desc = progress.tasks[task_id].description
                try:
                    future.result()
                    progress.advance(task_id)
                except Exception as exc:
                    progress.update(task_id, description=f"{task_desc} [failed]", advance=1)
                    console.print(f"[red]{task_desc} failed: {exc}[/]")
                    raise SystemExit(1)
    console.print("[green]Dependencies installed[/]")


def run_preflight(args, log_startup) -> None:
    """Run preflight checks unless skipped."""
    if args.skip_preflight:
        return
    results = preflight.run_preflight()
    failures: List[tuple[str, str]] = []
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
        raise SystemExit(1)


def perform_checks(
    args,
    rest: List[str],
    *,
    ensure_deps,
    ensure_target,
    ensure_wallet_cli,
    ensure_rpc,
    ensure_endpoints,
    ensure_cargo,
    run_quick_setup,
    log_startup,
    apply_env_overrides,
    load_config,
) -> Dict[str, Any]:
    """Run startup checks prior to launching."""
    disk_required = _disk_space_required_bytes(apply_env_overrides, load_config)

    disk_status = check_disk_space(disk_required, log_startup)
    internet_status = check_network(args, log_startup)

    (
        config_path,
        cfg_data,
        keypair_path,
        mnemonic_path,
        active_keypair,
        config_status,
        wallet_status,
    ) = ensure_configuration_and_wallet(args, ensure_wallet_cli, run_quick_setup)

    endpoint_status = check_endpoints(args, cfg_data, ensure_endpoints)

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
        return {"rest": rest, "summary_rows": [], "code": b_code or p_code}

    if args.diagnostics:
        from scripts import diagnostics

        diagnostics.main()
        return {"rest": rest, "summary_rows": [], "code": 0}

    if args.one_click:
        rest = ["--non-interactive", *rest]

    with contextlib.ExitStack() as stack:
        stack.enter_context(temporary_env("SOLHUNTER_SKIP_DEPS", "1"))
        if args.skip_setup or args.one_click:
            stack.enter_context(temporary_env("SOLHUNTER_SKIP_SETUP", "1"))
        if args.skip_preflight:
            stack.enter_context(temporary_env("SOLHUNTER_SKIP_PREFLIGHT", "1"))
        if args.no_diagnostics:
            stack.enter_context(temporary_env("SOLHUNTER_NO_DIAGNOSTICS", "1"))

        install_dependencies(args, ensure_deps, ensure_target)

        if sys.version_info < (3, 11):
            print(
                "Python 3.11 or higher is required. "
                "Please install Python 3.11 following the instructions in README.md."
            )
            return {"rest": rest, "summary_rows": [], "code": 1}

        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            print("Warning: running under Rosetta; Metal acceleration unavailable.")
            if not args.allow_rosetta:
                print("Use '--allow-rosetta' to continue anyway.")
                return {"rest": rest, "summary_rows": [], "code": 1}

        run_preflight(args, log_startup)

        if args.offline:
            rpc_status = "offline"
        elif args.skip_rpc_check:
            rpc_status = "skipped"
        else:
            ensure_rpc(warn_only=args.one_click)
            rpc_status = "reachable"
        from solhunter_zero.bootstrap import bootstrap

        with temporary_env("SOLHUNTER_SKIP_SETUP", "1"):
            bootstrap(one_click=args.one_click)

        gpu_env = device.initialize_gpu()
        gpu_device = gpu_env.get("SOLHUNTER_GPU_DEVICE", "unknown")
        rpc_url = os.environ.get(
            "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
        )

        log_startup(f"GPU device: {gpu_device}")
        log_startup(f"RPC endpoint: {rpc_url} ({rpc_status})")

        ensure_cargo()

    summary_rows = [
        ("Disk space", disk_status),
        ("Internet", internet_status),
        ("Configuration", str(config_status)),
        ("Wallet", str(wallet_status)),
        ("HTTP endpoints", endpoint_status),
    ]

    return {
        "summary_rows": summary_rows,
        "rest": rest,
        "config_path": config_path,
        "keypair_path": keypair_path,
        "mnemonic_path": mnemonic_path,
        "active_keypair": active_keypair,
    }
