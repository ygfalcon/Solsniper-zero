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
import webbrowser
from pathlib import Path
from typing import IO

from solhunter_zero.paths import ROOT
from solhunter_zero.logging_utils import log_startup  # noqa: E402

log_startup("start_all launched")

from solhunter_zero import device  # noqa: E402
from solhunter_zero.system import set_rayon_threads  # noqa: E402

device.ensure_gpu_env()
set_rayon_threads()

from solhunter_zero.config import (  # noqa: E402
    set_env_from_config,
    ensure_config_file,
    validate_env,
)
import solhunter_zero.config as config  # noqa: E402
from solhunter_zero import data_sync  # noqa: E402
from solhunter_zero.service_launcher import (  # noqa: E402
    start_depth_service,
    start_rl_daemon,
    wait_for_depth_ws,
)
from solhunter_zero.autopilot import _maybe_start_event_bus  # noqa: E402
from solhunter_zero.bootstrap_utils import ensure_cargo  # noqa: E402
import solhunter_zero.ui as ui  # noqa: E402
from solhunter_zero import bootstrap  # noqa: E402


class ProcessManager:
    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self.ws_threads: dict[str, threading.Thread] = {}
        self._stopped = False

    @staticmethod
    def _stream_stderr(pipe: IO[bytes]) -> None:
        for line in iter(pipe.readline, b""):
            sys.stderr.buffer.write(line)
        pipe.close()

    def start(
        self, cmd: list[str], *, stream_stderr: bool = False
    ) -> subprocess.Popen:
        env = os.environ.copy()
        for var in config.REQUIRED_ENV_VARS:
            val = os.getenv(var)
            if val is not None:
                env[var] = val
        stderr = subprocess.PIPE if stream_stderr else None
        proc = subprocess.Popen(cmd, env=env, stderr=stderr)
        self.procs.append(proc)
        if stream_stderr and proc.stderr is not None:
            threading.Thread(
                target=self._stream_stderr, args=(proc.stderr,), daemon=True
            ).start()
        return proc

    def stop_all(self, *_: object) -> None:
        if self._stopped:
            return
        self._stopped = True
        data_sync.stop_scheduler()
        for loop in (ui.rl_ws_loop, ui.event_ws_loop, ui.log_ws_loop):
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
        for thread in self.ws_threads.values():
            thread.join(timeout=1)
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 5
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.wait(deadline - time.time())
                except Exception:
                    p.kill()
        sys.exit(0)

    def monitor_processes(self) -> None:
        try:
            while any(p.poll() is None for p in self.procs):
                time.sleep(1)
        finally:
            self.stop_all()

    def __enter__(self) -> "ProcessManager":
        signal.signal(signal.SIGINT, self.stop_all)
        signal.signal(signal.SIGTERM, self.stop_all)
        logging.basicConfig(level=logging.INFO)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._stopped:
            self.stop_all()


def _wait_for_rl_daemon(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    """Wait briefly to ensure the RL daemon is running."""
    deadline = time.monotonic() + timeout
    ready_after = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"rl_daemon exited with code {proc.returncode}"
            )
        if time.monotonic() >= ready_after:
            return
        time.sleep(0.1)
    raise TimeoutError("rl_daemon startup timed out")


def launch_services(pm: ProcessManager) -> None:
    bootstrap.ensure_keypair()
    cfg = ensure_config_file()
    cfg_data = validate_env(config.REQUIRED_ENV_VARS, cfg)
    set_env_from_config(cfg_data)
    import solhunter_zero.config as config  # noqa: E402
    config.reload_active_config()
    interval = float(
        cfg_data.get(
            "offline_data_interval", os.getenv("OFFLINE_DATA_INTERVAL", "3600")
        )
    )
    db_path_val = cfg_data.get("rl_db_path", "offline_data.db")
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

    _maybe_start_event_bus(cfg_data)

    ensure_cargo()
    depth_proc = start_depth_service(cfg, stream_stderr=True)
    pm.procs.append(depth_proc)
    os.environ["DEPTH_SERVICE"] = "false"
    rl_proc = start_rl_daemon()
    pm.procs.append(rl_proc)
    addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
    port = int(os.getenv("DEPTH_WS_PORT", "8766"))
    deadline = time.monotonic() + 30.0
    wait_for_depth_ws(addr, port, deadline, depth_proc)
    _wait_for_rl_daemon(rl_proc)

    main_cmd = [sys.executable, "-m", "solhunter_zero.main"]
    if cfg:
        main_cmd += ["--config", cfg]
    pm.start(main_cmd)


def launch_ui(pm: ProcessManager) -> None:
    def _run_ui() -> None:
        app = ui.create_app()
        pm.ws_threads = ui.start_websockets()
        app.run()

    thread = threading.Thread(target=_run_ui, daemon=True)
    thread.start()
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


def main() -> None:
    with ProcessManager() as pm:
        launch_services(pm)
        launch_ui(pm)
        pm.monitor_processes()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
        from solhunter_zero import autopilot

        autopilot.main()
        sys.exit(0)
    else:
        main()
