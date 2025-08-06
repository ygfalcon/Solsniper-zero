#!/usr/bin/env python3
"""Launcher for SolHunter Zero.

This module consolidates the behaviour of the previous ``start.command`` and
``run.sh`` scripts into a cross platform Python entry point.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _ensure_config() -> None:
    cfg = ROOT / "config.toml"
    if not cfg.exists():
        shutil.copy2(ROOT / "config.example.toml", cfg)
        print("Created default config.toml from config.example.toml")


def _check_python() -> None:
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required.", file=sys.stderr)
        raise SystemExit(1)


def _set_rayon_threads() -> None:
    if os.getenv("RAYON_NUM_THREADS"):
        return
    for cmd in ("nproc", ("getconf", "_NPROCESSORS_ONLN"), ("sysctl", "-n", "hw.ncpu")):
        if isinstance(cmd, tuple):
            if shutil.which(cmd[0]):
                os.environ["RAYON_NUM_THREADS"] = subprocess.check_output(cmd).decode().strip()
                return
        elif shutil.which(cmd):
            os.environ["RAYON_NUM_THREADS"] = subprocess.check_output([cmd]).decode().strip()
            return
    os.environ["RAYON_NUM_THREADS"] = str(os.cpu_count() or 1)


def _gpu_setup() -> None:
    from solhunter_zero import device

    if device.detect_gpu():
        os.environ.setdefault("GPU_MEMORY_INDEX", "1")
        if platform.system() == "Darwin":
            os.environ.setdefault("TORCH_DEVICE", "mps")
    else:
        print("No GPU detected; using CPU mode")


def _check_rpc() -> None:
    url = os.environ.get("SOLANA_RPC_URL")
    if not url:
        return
    if url.startswith("ws://"):
        url = "http://" + url[5:]
    elif url.startswith("wss://"):
        url = "https://" + url[6:]
    req = urllib.request.Request(url, method="HEAD")
    for attempt in range(3):
        try:
            urllib.request.urlopen(req, timeout=5)
            return
        except Exception as exc:  # pragma: no cover - network failures
            if attempt == 2:
                print(
                    f"Error: SOLANA_RPC_URL '{os.environ['SOLANA_RPC_URL']}' is unreachable: {exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            wait = 2 ** attempt
            print(
                f"Attempt {attempt + 1} failed to reach SOLANA_RPC_URL at {url}: {exc}.",
                f" Retrying in {wait} seconds...",
                file=sys.stderr,
            )
            time.sleep(wait)


def _build_depth_service() -> None:
    subprocess.check_call(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(ROOT / "depth_service" / "Cargo.toml"),
            "--release",
        ]
    )


def _run(cmd: Iterable[str]) -> None:
    subprocess.check_call(list(cmd))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--start-all", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args, rest = parser.parse_known_args(argv)

    _check_python()
    os.environ.setdefault("DEPTH_SERVICE", "true")
    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    _ensure_config()

    from scripts import startup  # heavy import, done lazily

    startup.ensure_deps()
    startup.ensure_cargo()
    startup.ensure_route_ffi()
    _build_depth_service()
    _check_rpc()
    _set_rayon_threads()
    _gpu_setup()

    agg_proc: subprocess.Popen[str] | None = None
    log_path: str | None = None
    if not args.no_metrics:
        log_fd, log_path = tempfile.mkstemp()
        os.close(log_fd)
        agg_proc = subprocess.Popen(
            [sys.executable, "-m", "solhunter_zero.metrics_aggregator"],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(1)
        if agg_proc.poll() is not None:
            print("metrics_aggregator failed to start", file=sys.stderr)
            with open(log_path) as fh:
                print(fh.read(), file=sys.stderr)
            return 1

    try:
        if args.daemon:
            _run([sys.executable, "-m", "solhunter_zero.train_cli", "--daemon", *rest])
        elif args.start_all or (not args.auto and not rest and platform.system() == "Darwin"):
            _run([sys.executable, "scripts/start_all.py", "autopilot"])
        else:
            cmd = [sys.executable, "-m", "solhunter_zero.main"]
            if args.auto or not rest:
                cmd.append("--auto")
            cmd.extend(rest)
            _run(cmd)
    finally:
        if agg_proc:
            agg_proc.terminate()
            try:
                agg_proc.wait(timeout=2)
            except Exception:  # pragma: no cover - best effort cleanup
                agg_proc.kill()
        if log_path and os.path.exists(log_path):
            os.unlink(log_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
