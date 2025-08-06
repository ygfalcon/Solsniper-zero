import os
import sys
import subprocess
import threading
import time
import socket
import logging
from pathlib import Path
from typing import IO

from solhunter_zero.paths import ROOT
logger = logging.getLogger(__name__)


def _stream_stderr(pipe: IO[bytes]) -> None:
    for line in iter(pipe.readline, b""):
        sys.stderr.buffer.write(line)
    pipe.close()


def start_depth_service(
    cfg_path: str | None = None, *, stream_stderr: bool = False
) -> subprocess.Popen:
    """Start the depth_service binary, building it if needed."""
    depth_bin = ROOT / "target" / "release" / "depth_service"
    if not depth_bin.exists() or not os.access(depth_bin, os.X_OK):
        logger.info("depth_service binary not found, building with cargo...")
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "build",
                    "--manifest-path",
                    str(ROOT / "depth_service" / "Cargo.toml"),
                    "--release",
                ]
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "cargo is not installed. Please run "
                "'cargo build --manifest-path depth_service/Cargo.toml "
                "--release'"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build depth_service. Please run "
                "'cargo build --manifest-path depth_service/Cargo.toml "
                "--release' manually."
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
