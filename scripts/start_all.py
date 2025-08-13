#!/usr/bin/env python3
"""Launch depth_service, RL daemon and trading bot."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import bootstrap utilities even when the repository has not
# been installed as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solhunter_zero.bootstrap_utils import ensure_venv, prepend_repo_root  # noqa: E402

# Re-exec inside the local virtual environment if necessary.
ensure_venv(None)

# When running inside the project's virtual environment, make the repository
# importable without requiring ``pip install -e .``.
if Path(sys.prefix).resolve() == REPO_ROOT / ".venv":
    prepend_repo_root()

import os
import signal
import subprocess
import time
import threading
import logging
import webbrowser
import socket
import urllib.parse
from typing import IO

from solhunter_zero.paths import ROOT  # noqa: E402
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
    initialize_event_bus,
)
import solhunter_zero.config as config  # noqa: E402
from solhunter_zero import data_sync  # noqa: E402
from solhunter_zero.service_launcher import (  # noqa: E402
    start_depth_service,
    start_rl_daemon,
    wait_for_depth_ws,
)
from solhunter_zero.autopilot import (  # noqa: E402
    _maybe_start_event_bus,
    shutdown_event_bus,
)
from solhunter_zero.bootstrap_utils import ensure_cargo  # noqa: E402
import solhunter_zero.ui as ui  # noqa: E402
from solhunter_zero import bootstrap  # noqa: E402


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _check_redis_connection() -> None:
    """Log a helpful message when the Redis broker is unreachable."""
    url = (
        os.getenv("EVENT_BUS_URL")
        or os.getenv("BROKER_URL")
        or "redis://127.0.0.1:6379"
    )
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"redis", "rediss"}:
        return
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    if _is_port_open(host, port):
        return
    logging.error(
        "Failed to connect to Redis at %s:%s. "
        "Start redis-server or set EVENT_BUS_URL.",
        host,
        port,
    )

class ProcessManager:
    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self.ws_threads: dict[str, threading.Thread] = {}
        self.redis_proc: subprocess.Popen | None = None
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

    def stop_all(self, *_: object, exit_code: int = 0) -> None:
        if self._stopped:
            return
        self._stopped = True
        data_sync.stop_scheduler()
        shutdown_event_bus()
        for loop in (ui.rl_ws_loop, ui.event_ws_loop, ui.log_ws_loop):
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
        for thread in self.ws_threads.values():
            thread.join(timeout=1)
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        if self.redis_proc and self.redis_proc.poll() is None:
            self.redis_proc.terminate()
        deadline = time.time() + 5
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.wait(deadline - time.time())
                except Exception:
                    p.kill()
        if self.redis_proc and self.redis_proc.poll() is None:
            try:
                self.redis_proc.wait(deadline - time.time())
            except Exception:
                self.redis_proc.kill()
        sys.exit(exit_code)

    def monitor_processes(self) -> None:
        try:
            if not self.procs:
                logging.error("No child processes were started")
                self.stop_all(exit_code=1)
                return
            while True:
                running = [p for p in self.procs if p.poll() is None]
                if not running:
                    logging.error("All child processes exited")
                    self.stop_all(exit_code=1)
                    return
                time.sleep(1)
                for p in running:
                    rc = p.poll()
                    if rc not in (None, 0):
                        logging.error(
                            "Process %s exited with code %s", p.args, rc
                        )
                        self.stop_all(exit_code=1)
                        return
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
    """Wait for the RL daemon to emit a heartbeat event."""

    from solhunter_zero import event_bus

    deadline = time.monotonic() + timeout
    hb_received = threading.Event()

    def _on_hb(payload: object) -> None:
        service = (
            payload.get("service") if isinstance(payload, dict) else getattr(payload, "service", None)
        )
        if service == "rl_daemon":
            hb_received.set()

    unsub = event_bus.subscribe("heartbeat", _on_hb)
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"rl_daemon exited with code {proc.returncode}"
                )
            if hb_received.wait(0.1):
                return
        raise TimeoutError("rl_daemon startup timed out")
    finally:
        unsub()


def launch_services(pm: ProcessManager) -> None:
    # Ensure optional dependencies are installed and environment checks are
    # performed before launching any additional processes.
    bootstrap.bootstrap(one_click=True)
    bootstrap.ensure_keypair()
    cfg = ensure_config_file()
    import solhunter_zero.config as config  # noqa: E402
    cfg_data = validate_env(config.REQUIRED_ENV_VARS, cfg)
    set_env_from_config(cfg_data)
    config.reload_active_config()
    _check_redis_connection()
    config.get_solana_ws_url()
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
    urls: list[str] = []
    url = os.getenv("BROKER_URL")
    if url:
        urls.append(url)
    more = os.getenv("BROKER_URLS")
    if more:
        urls.extend(u.strip() for u in more.split(",") if u.strip())
    needs_redis = any(u.startswith("redis://") for u in urls)
    if needs_redis and not _is_port_open("127.0.0.1", 6379):
        pm.redis_proc = subprocess.Popen(["redis-server"])
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pm.redis_proc.poll() is not None:
                raise RuntimeError("redis-server exited during startup")
            if _is_port_open("127.0.0.1", 6379):
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("redis-server startup timed out")

    _maybe_start_event_bus(cfg_data)
    initialize_event_bus()

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

    main_cmd = [sys.executable, "-m", "solhunter_zero.main", "--auto"]
    if cfg:
        main_cmd += ["--config", cfg]
    pm.start(main_cmd)


def launch_ui(pm: ProcessManager) -> None:
    def _run_ui() -> None:
        try:
            app = ui.create_app()
            pm.ws_threads = ui.start_websockets()
            app.run()
        except Exception as exc:  # noqa: BLE001
            log_startup(f"UI initialization failed: {exc}")
            pm.stop_all()

    thread = threading.Thread(target=_run_ui, daemon=True)
    thread.start()
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


def main() -> None:
    with ProcessManager() as pm:
        try:
            launch_services(pm)
        except Exception:
            shutdown_event_bus()
            raise
        launch_ui(pm)
        pm.monitor_processes()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
        from solhunter_zero import autopilot

        autopilot.main()
        sys.exit(0)
    else:
        main()
