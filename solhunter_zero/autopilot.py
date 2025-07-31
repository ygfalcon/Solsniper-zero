from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import asyncio
from pathlib import Path

from . import wallet, data_sync, main
from .config import CONFIG_DIR, get_active_config_name, load_config, apply_env_overrides

ROOT = Path(__file__).resolve().parent.parent
PROCS: list[subprocess.Popen] = []


def _start(cmd: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)
    PROCS.append(proc)
    return proc


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
    active = wallet.get_active_keypair_name()
    if active is None:
        keys = wallet.list_keypairs()
        if len(keys) == 1:
            wallet.select_keypair(keys[0])
            active = keys[0]
    if active and not os.getenv("KEYPAIR_PATH"):
        os.environ["KEYPAIR_PATH"] = os.path.join(wallet.KEYPAIR_DIR, active + ".json")


def _get_config() -> tuple[str | None, dict]:
    name = get_active_config_name()
    cfg_path: str | None = None
    if name:
        path = os.path.join(CONFIG_DIR, name)
        if os.path.isfile(path):
            cfg_path = path
    else:
        preset = Path(ROOT / "config.highrisk.toml")
        if preset.is_file():
            cfg_path = str(preset)
    cfg: dict = {}
    if cfg_path:
        cfg = apply_env_overrides(load_config(cfg_path))
    return cfg_path, cfg


async def _wait_depth(addr: str, port: int, deadline: float) -> None:
    """Wait for the depth_service websocket to accept connections."""
    while True:
        try:
            reader, writer = await asyncio.open_connection(addr, port)
        except OSError:
            if time.monotonic() > deadline:
                print("depth_service websocket timed out", file=sys.stderr)
                _stop_all()
            await asyncio.sleep(0.1)
        else:
            writer.close()
            await writer.wait_closed()
            break


def main() -> None:
    os.chdir(ROOT)
    signal.signal(signal.SIGINT, _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    _ensure_keypair()

    cfg_path, cfg = _get_config()
    interval = float(cfg.get("offline_data_interval", os.getenv("OFFLINE_DATA_INTERVAL", "3600")))
    db_path = cfg.get("rl_db_path", "offline_data.db")
    data_sync.start_scheduler(interval=interval, db_path=db_path)

    cmd = ["./target/release/depth_service"]
    if cfg_path:
        cmd += ["--config", cfg_path]
    _start(cmd)
    _start([sys.executable, "scripts/run_rl_daemon.py"])

    addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
    port = int(os.getenv("DEPTH_WS_PORT", "8765"))
    deadline = time.monotonic() + 30.0
    asyncio.run(_wait_depth(addr, port, deadline))

    main.run_auto()
    _stop_all()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
