#!/usr/bin/env python3
"""Launch depth_service, RL daemon and trading bot."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

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


def start(cmd: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)
    PROCS.append(proc)
    return proc


def stop_all(*_: object) -> None:
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

# Launch depth service and RL daemon first
start(["./target/release/depth_service"])
start([sys.executable, "scripts/run_rl_daemon.py"])

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

start([sys.executable, "-m", "solhunter_zero.main"])

try:
    while any(p.poll() is None for p in PROCS):
        time.sleep(1)
finally:
    stop_all()
