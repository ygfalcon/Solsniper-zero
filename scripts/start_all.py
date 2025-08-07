#!/usr/bin/env python3
"""Launch depth_service, RL daemon and trading bot."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from solhunter_zero.logging_utils import log_startup, rotate_startup_log  # noqa: E402
from solhunter_zero import env  # noqa: E402

rotate_startup_log()
env.load_env_file(ROOT / ".env")
os.chdir(ROOT)
log_startup("start_all launched")

from solhunter_zero import device  # noqa: E402
from solhunter_zero.system import set_rayon_threads  # noqa: E402

device.ensure_gpu_env()
set_rayon_threads()

from solhunter_zero.config import (  # noqa: E402
    set_env_from_config,
    ensure_config_file,
    validate_env,
    REQUIRED_ENV_VARS,
)
from solhunter_zero import data_sync  # noqa: E402
from solhunter_zero.service_launcher import (  # noqa: E402
    start_depth_service,
    start_rl_daemon,
    wait_for_depth_ws,
)

if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
    from solhunter_zero import autopilot

    autopilot.main()
    raise SystemExit

PROCS: list[subprocess.Popen] = []


ENV_VARS = REQUIRED_ENV_VARS + (
    "DEPTH_SERVICE_SOCKET",
    "DEPTH_MMAP_PATH",
    "DEPTH_WS_ADDR",
    "DEPTH_WS_PORT",
)


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
        threading.Thread(
            target=_stream_stderr, args=(proc.stderr,), daemon=True
        ).start()
    return proc


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
logging.basicConfig(level=logging.INFO)

cfg = ensure_config_file()
cfg_data = validate_env(ENV_VARS, cfg)
set_env_from_config(cfg_data)

# Launch depth service and RL daemon first
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

try:
    depth_proc = start_depth_service(cfg, stream_stderr=True)
    PROCS.append(depth_proc)
    PROCS.append(start_rl_daemon())
    addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
    port = int(os.getenv("DEPTH_WS_PORT", "8765"))
    deadline = time.monotonic() + 30.0
    wait_for_depth_ws(addr, port, deadline, depth_proc)
except Exception as exc:
    logging.error(str(exc))
    stop_all()

main_cmd = [sys.executable, "-m", "solhunter_zero.main"]
if cfg:
    main_cmd += ["--config", cfg]
start(main_cmd)

try:
    while any(p.poll() is None for p in PROCS):
        time.sleep(1)
finally:
    stop_all()
