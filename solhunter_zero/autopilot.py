from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import data_sync
from . import main as main_module
from . import wallet
from .config import CONFIG_DIR, apply_env_overrides, get_active_config_name, load_config
from .paths import ROOT
from .service_launcher import start_depth_service, start_rl_daemon, wait_for_depth_ws

PROCS: list[subprocess.Popen] = []


def _stop_all(*_: object) -> None:
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
    sys.exit(0)


def _ensure_keypair() -> None:
    try:
        info = wallet.setup_default_keypair()
        path = os.path.join(wallet.KEYPAIR_DIR, info.name + ".json")
        os.environ["KEYPAIR_PATH"] = path
        print(f"Using keypair: {info.name}")
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC "
            "environment variable.",
            file=sys.stderr,
        )
        raise SystemExit(1)


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
    os.chdir(ROOT)
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
    db_path = cfg.get("rl_db_path", "offline_data.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(db_path, "a"):
            pass
    except OSError as exc:
        print(f"Cannot write to {db_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    data_sync.start_scheduler(interval=interval, db_path=db_path)

    try:
        depth_proc = start_depth_service(cfg_path)
        PROCS.append(depth_proc)
        PROCS.append(start_rl_daemon())

        addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
        port = int(os.getenv("DEPTH_WS_PORT", "8765"))
        deadline = time.monotonic() + 30.0
        wait_for_depth_ws(addr, port, deadline, depth_proc)
    except Exception as exc:
        logging.error(str(exc))
        _stop_all()

    main_module.run_auto()
    _stop_all()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
