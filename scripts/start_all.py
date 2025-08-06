#!/usr/bin/env python3
"""Launch depth_service, RL daemon and trading bot."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import time
import socket
import threading
from pathlib import Path

from solhunter_zero.config import load_config, apply_env_overrides
from solhunter_zero import data_sync

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
    from solhunter_zero import autopilot

    autopilot.main()
    raise SystemExit

PROCS: list[subprocess.Popen] = []
# keep track of log file handles so we can close them on shutdown
LOG_HANDLES: list[tuple[io.IOBase, io.IOBase]] = []


ENV_VARS = [
    "EVENT_BUS_URL",
    "SOLANA_RPC_URL",
    "SOLANA_KEYPAIR",
    "DEPTH_SERVICE_SOCKET",
    "DEPTH_MMAP_PATH",
    "DEPTH_WS_ADDR",
    "DEPTH_WS_PORT",
]


def start(
    cmd: list[str],
    name: str | None = None,
    stream_stderr: bool = True,
) -> subprocess.Popen:
    """Launch a subprocess with logging.

    Stdout and stderr are redirected to ``logs/<name>-stdout.log`` and
    ``logs/<name>-stderr.log`` respectively.  The log paths are printed so
    users can inspect startup errors.  If ``stream_stderr`` is true the stderr
    stream is forwarded to the console in real time while still being written
    to the log file.
    """

    env = os.environ.copy()
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    if name is None:
        name = Path(cmd[0]).stem

    stdout_path = logs_dir / f"{name}-stdout.log"
    stderr_path = logs_dir / f"{name}-stderr.log"

    stdout_f = open(stdout_path, "wb")
    stderr_f = open(stderr_path, "wb")

    if stream_stderr:
        proc = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=subprocess.PIPE)

        def _forward_stderr() -> None:
            assert proc.stderr is not None  # for type checkers
            for line in proc.stderr:
                stderr_f.write(line)
                stderr_f.flush()
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.buffer.flush()
                except Exception:
                    pass
            proc.stderr.close()
            stderr_f.close()

        threading.Thread(target=_forward_stderr, daemon=True).start()
    else:
        proc = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f)

    PROCS.append(proc)
    LOG_HANDLES.append((stdout_f, stderr_f))
    print(f"Started {' '.join(cmd)}; stdout -> {stdout_path}, stderr -> {stderr_path}")
    return proc


def get_config_file() -> str | None:
    path = os.getenv("SOLHUNTER_CONFIG")
    if path:
        return path
    cfg_dir = Path(os.getenv("CONFIG_DIR", "configs"))
    active = cfg_dir / "active"
    if active.is_file():
        name = active.read_text().strip()
        cfg = cfg_dir / name
        if cfg.is_file():
            return str(cfg)
    for name in ("config.toml", "config.yaml", "config.yml"):
        if Path(name).is_file():
            return name
    return None


def stop_all(*_: object) -> None:
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
    for stdout_f, stderr_f in LOG_HANDLES:
        try:
            stdout_f.close()
        except Exception:
            pass
        try:
            stderr_f.close()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

# Launch depth service and RL daemon first
cfg = get_config_file()
cfg_data = {}
if cfg:
    cfg_data = apply_env_overrides(load_config(cfg))
interval = float(cfg_data.get("offline_data_interval", os.getenv("OFFLINE_DATA_INTERVAL", "3600")))
db_path = cfg_data.get("rl_db_path", "offline_data.db")
data_sync.start_scheduler(interval=interval, db_path=db_path)
cmd = ["./target/release/depth_service"]
if cfg:
    cmd += ["--config", cfg]
start(cmd, name="depth_service")
start([sys.executable, "scripts/run_rl_daemon.py"], name="rl_daemon")

# Wait for the websocket to come online before starting the bot
addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
port = int(os.getenv("DEPTH_WS_PORT", "8765"))
deadline = time.monotonic() + 30.0
while True:
    try:
        with socket.create_connection((addr, port), timeout=1):
            break
    except OSError:
        if time.monotonic() > deadline:
            print("depth_service websocket timed out", file=sys.stderr)
            stop_all()
        time.sleep(0.1)

main_cmd = [sys.executable, "-m", "solhunter_zero.main"]
if cfg:
    main_cmd += ["--config", cfg]
start(main_cmd, name="trading_bot")

try:
    while any(p.poll() is None for p in PROCS):
        time.sleep(1)
finally:
    stop_all()
