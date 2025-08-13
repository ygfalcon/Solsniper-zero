from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import logging
from pathlib import Path

from . import wallet, data_sync, main as main_module
from .config import (
    CONFIG_DIR,
    get_active_config_name,
    load_config,
    apply_env_overrides,
)
from .service_launcher import (
    start_depth_service,
    start_rl_daemon,
    wait_for_depth_ws,
)
from .paths import ROOT
PROCS: list[subprocess.Popen] = []


def _stop_all(*_: object, exit_code: int = 0) -> None:
    data_sync.stop_scheduler()
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 5
    for p in PROCS:
        if p.poll() is None:
            try:
                p.wait(deadline - time.time())
            except Exception:
                p.kill()
    sys.exit(exit_code)


def _ensure_keypair() -> None:
    from solhunter_zero.startup_checks import ensure_wallet_cli, run_quick_setup

    ensure_wallet_cli()
    try:
        info = wallet.setup_default_keypair()
    except Exception as exc:
        print(f"Wallet interaction failed: {exc}", file=sys.stderr)
        print("Attempting quick non-interactive setup...", file=sys.stderr)
        try:
            run_quick_setup()
            info = wallet.setup_default_keypair()
        except Exception as exc2:
            print(
                f"Wallet interaction failed: {exc2}\n"
                "Run 'solhunter-wallet' manually or set the MNEMONIC "
                "environment variable.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    path = os.path.join(wallet.KEYPAIR_DIR, info.name + ".json")
    os.environ["KEYPAIR_PATH"] = path
    print(f"Using keypair: {info.name}")


def _get_config() -> tuple[str | None, dict]:
    name = get_active_config_name()
    cfg_path: str | None = None
    if name:
        path = os.path.join(CONFIG_DIR, name)
        if os.path.isfile(path):
            cfg_path = path
    else:
        preset = Path(ROOT / "config" / "default.toml")
        if preset.is_file():
            cfg_path = str(preset)
    cfg: dict = {}
    if cfg_path:
        cfg = apply_env_overrides(load_config(cfg_path))
    return cfg_path, cfg


def main() -> None:
    signal.signal(signal.SIGINT, _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    logging.basicConfig(level=logging.INFO)
    from . import device

    device.initialize_gpu()

    _ensure_keypair()

    cfg_path, cfg = _get_config()
    interval = float(
        cfg.get(
            "offline_data_interval",
            os.getenv("OFFLINE_DATA_INTERVAL", "3600"),
        )
    )
    db_path_val = cfg.get("rl_db_path", "offline_data.db")
    db_path = Path(db_path_val)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(db_path, "a"):
            pass
    except OSError as exc:
        print(f"Cannot write to {db_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    data_sync.start_scheduler(interval=interval, db_path=str(db_path))

    try:
        depth_proc = start_depth_service(cfg_path)
        PROCS.append(depth_proc)
        PROCS.append(start_rl_daemon())

        addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
        port = int(os.getenv("DEPTH_WS_PORT", "8766"))
        deadline = time.monotonic() + 30.0
        wait_for_depth_ws(addr, port, deadline, depth_proc)
    except Exception as exc:
        logging.error(str(exc))
        _stop_all(exit_code=1)

    exit_code = 0
    try:
        main_module.run_auto()
    except Exception:
        logging.exception("run_auto failed")
        exit_code = 1
    finally:
        _stop_all(exit_code=exit_code)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
