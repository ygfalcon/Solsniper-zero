#!/usr/bin/env python3
"""Launch depth_service, RL daemon and trading bot."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import socket
import threading
from pathlib import Path
from typing import IO
from solhunter_zero.config import (
    ENV_VARS as CONFIG_ENV_VARS,
    apply_env_overrides,
    load_config,
    set_env_from_config,
)
from solhunter_zero import data_sync
from scripts.rust_utils import build_depth_service, build_route_ffi, ensure_cargo

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
    from solhunter_zero import autopilot

    autopilot.main()
    raise SystemExit

PROCS: list[subprocess.Popen] = []


ENV_VARS = [
    "EVENT_BUS_URL",
    "SOLANA_RPC_URL",
    "SOLANA_KEYPAIR",
    "DEPTH_SERVICE_SOCKET",
    "DEPTH_MMAP_PATH",
    "DEPTH_WS_ADDR",
    "DEPTH_WS_PORT",
]


def validate_env() -> None:
    cfg_data = {}
    cfg = get_config_file()
    if cfg:
        cfg_data = apply_env_overrides(load_config(cfg))
    env_to_key = {v: k for k, v in CONFIG_ENV_VARS.items()}
    missing: list[str] = []
    for name in ENV_VARS:
        if not os.getenv(name):
            val = None
            key = env_to_key.get(name)
            if key:
                val = cfg_data.get(key)
            if val is None:
                val = cfg_data.get(name)
            if val is not None:
                os.environ[name] = str(val)
            if not os.getenv(name):
                missing.append(name)
    if missing:
        for name in missing:
            print(f"Required env var {name} is not set", file=sys.stderr)
        sys.exit(1)


def _stream_stderr(pipe: IO[bytes]) -> None:
    for line in iter(pipe.readline, b""):
        sys.stderr.buffer.write(line)
    pipe.close()


def start(cmd: list[str], *, stream_stderr: bool = False) -> subprocess.Popen:
    env = os.environ.copy()
    for var in ENV_VARS:
        val = os.getenv(var)
        if val is not None:
            env[var] = val
    stderr = subprocess.PIPE if stream_stderr else None
    proc = subprocess.Popen(cmd, env=env, stderr=stderr)
    PROCS.append(proc)
    if stream_stderr and proc.stderr is not None:
        threading.Thread(target=_stream_stderr, args=(proc.stderr,), daemon=True).start()
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
    # If no config is found, generate a default one via quick_setup
    try:
        from scripts import quick_setup

        # Populate config with defaults
        quick_setup.main(["--auto"])
    except Exception as exc:
        print(f"Failed to generate default config: {exc}", file=sys.stderr)
        return None
    # Re-run lookup after generation
    return get_config_file()


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
    sys.exit(0)


signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

validate_env()

# Launch depth service and RL daemon first
cfg = get_config_file()
cfg_data = {}
if cfg:
    cfg_data = apply_env_overrides(load_config(cfg))
    set_env_from_config(cfg_data)
interval = float(
    cfg_data.get(
        "offline_data_interval", os.getenv("OFFLINE_DATA_INTERVAL", "3600")
    )
)
db_path = cfg_data.get("rl_db_path", "offline_data.db")
Path(db_path).parent.mkdir(parents=True, exist_ok=True)
try:
    with open(db_path, "a"):
        pass
except OSError as exc:
    print(f"Cannot write to {db_path}: {exc}", file=sys.stderr)
    sys.exit(1)
data_sync.start_scheduler(interval=interval, db_path=db_path)
ensure_cargo()
build_route_ffi()
depth_bin = build_depth_service()
cmd = [str(depth_bin)]
if cfg:
    cmd += ["--config", cfg]
depth_proc = start(cmd, stream_stderr=True)
start([sys.executable, "scripts/run_rl_daemon.py"])

# Wait for the websocket to come online before starting the bot
addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
port = int(os.getenv("DEPTH_WS_PORT", "8765"))
deadline = time.monotonic() + 30.0
while True:
    if depth_proc.poll() is not None:
        print(
            f"depth_service exited with code {depth_proc.returncode}",
            file=sys.stderr,
        )
        stop_all()
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
start(main_cmd)

try:
    while any(p.poll() is None for p in PROCS):
        time.sleep(1)
finally:
    stop_all()
