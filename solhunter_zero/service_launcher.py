import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Sequence

from .paths import ROOT
from .cache_paths import CARGO_MARKER
logger = logging.getLogger(__name__)


def _stream_stderr(pipe: IO[bytes]) -> None:
    for line in iter(pipe.readline, b""):
        sys.stderr.buffer.write(line)
    pipe.close()


def _ensure_cargo() -> None:
    """Ensure the Rust toolchain is installed via rustup."""
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if shutil.which("cargo") is not None:
        return
    if CARGO_MARKER.exists():
        raise RuntimeError(
            "Rust toolchain previously installed but 'cargo' was not found. "
            "Ensure ~/.cargo/bin is in your PATH or remove the cache marker and rerun."
        )
    CARGO_MARKER.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Installing Rust toolchain via rustup...")
    try:
        subprocess.run(["rustup-init", "-y"], check=True)
    except FileNotFoundError as exc:  # pragma: no cover - rustup missing is rare
        raise RuntimeError(
            "rustup-init not found. Please install rustup from https://rustup.rs/."
        ) from exc
    CARGO_MARKER.touch()
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo installation failed; ensure ~/.cargo/bin is in PATH")


def start_depth_service(
    cfg_path: str | None = None, *, stream_stderr: bool = False
) -> subprocess.Popen:
    """Start the depth_service binary, building it if needed."""
    depth_bin = ROOT / "target" / "release" / "depth_service"
    if not depth_bin.exists() or not os.access(depth_bin, os.X_OK):
        _ensure_cargo()
        logger.info("depth_service binary not found, building with cargo...")
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(ROOT / "depth_service" / "Cargo.toml"),
                "--release",
            ],
            check=True,
        )
        if not depth_bin.exists() or not os.access(depth_bin, os.X_OK):
            raise RuntimeError(
                "depth_service binary missing or not executable after build. "
                "Please run 'cargo build --manifest-path "
                "depth_service/Cargo.toml --release'."
            )
    cmd = [str(depth_bin)]
    if cfg_path:
        cmd += ["--config", cfg_path]
    env = os.environ.copy()
    stderr = subprocess.PIPE if stream_stderr else None
    proc = subprocess.Popen(cmd, env=env, stderr=stderr)
    if stream_stderr and proc.stderr is not None:
        threading.Thread(
            target=_stream_stderr, args=(proc.stderr,), daemon=True
        ).start()
    return proc


def start_rl_daemon() -> subprocess.Popen:
    """Start the reinforcement learning daemon."""
    env = os.environ.copy()
    cmd = [sys.executable, "scripts/run_rl_daemon.py"]
    return subprocess.Popen(cmd, env=env)


def wait_for_depth_ws(
    addr: str,
    port: int,
    deadline: float,
    depth_proc: subprocess.Popen | None = None,
) -> None:
    """Wait for the depth_service websocket to accept connections."""
    while True:
        if depth_proc is not None and depth_proc.poll() is not None:
            raise RuntimeError(
                f"depth_service exited with code {depth_proc.returncode}"
            )
        try:
            with socket.create_connection((addr, port), timeout=1):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("depth_service websocket timed out")
            time.sleep(0.1)


ENV_VARS = (
    "DEX_BASE_URL",
    "DEPTH_SERVICE_SOCKET",
    "DEPTH_MMAP_PATH",
    "DEPTH_WS_ADDR",
    "DEPTH_WS_PORT",
)


def start_background_services(cfg_path: str | None = None) -> Sequence[subprocess.Popen]:
    """Launch depth_service and RL daemon and start data sync scheduler."""
    from .config import set_env_from_config, ensure_config_file, validate_env
    from . import data_sync

    cfg = ensure_config_file(cfg_path)
    cfg_data = validate_env(ENV_VARS, cfg)
    set_env_from_config(cfg_data)

    interval = float(
        cfg_data.get(
            "offline_data_interval", os.getenv("OFFLINE_DATA_INTERVAL", "3600")
        )
    )
    db_path = cfg_data.get("rl_db_path", "offline_data.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    data_sync.start_scheduler(interval=interval, db_path=db_path)

    depth_proc = start_depth_service(cfg, stream_stderr=True)
    rl_proc = start_rl_daemon()
    addr = os.getenv("DEPTH_WS_ADDR", "127.0.0.1")
    port = int(os.getenv("DEPTH_WS_PORT", "8765"))
    deadline = time.monotonic() + 30.0
    wait_for_depth_ws(addr, port, deadline, depth_proc)
    return [depth_proc, rl_proc]


def stop_background_services(procs: Sequence[subprocess.Popen]) -> None:
    """Terminate running background service processes."""
    from . import data_sync

    data_sync.stop_scheduler()
    for p in procs:
        if p and p.poll() is None:
            p.terminate()
    deadline = time.time() + 5
    for p in procs:
        if p and p.poll() is None:
            try:
                p.wait(deadline - time.time())
            except Exception:
                p.kill()
