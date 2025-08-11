"""Background service utilities for SolHunter Zero."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import time
from pathlib import Path


# Manifest for the optional Rust ``depth_service`` companion process.
_SERVICE_MANIFEST = (
    Path(__file__).resolve().parent.parent / "depth_service" / "Cargo.toml"
)


def start_depth_service(cfg: dict) -> subprocess.Popen | None:
    """Launch the Rust ``depth_service`` if enabled."""
    if not cfg.get("depth_service"):
        return None

    args = [
        "cargo",
        "run",
        "--manifest-path",
        str(_SERVICE_MANIFEST),
        "--release",
        "--",
    ]

    def add(flag: str, key: str) -> None:
        val = os.getenv(key.upper()) or cfg.get(key)
        if val:
            args.extend([flag, str(val)])

    add("--raydium", "raydium_ws_url")
    add("--orca", "orca_ws_url")
    add("--phoenix", "phoenix_ws_url")
    add("--meteora", "meteora_ws_url")
    add("--jupiter", "jupiter_ws_url")
    add("--serum", "serum_ws_url")

    rpc = os.getenv("SOLANA_RPC_URL") or cfg.get("solana_rpc_url")
    if rpc:
        args.extend(["--rpc", rpc])
    keypair = os.getenv("SOLANA_KEYPAIR") or os.getenv("KEYPAIR_PATH")
    if keypair:
        args.extend(["--keypair", keypair])

    socket_path = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")
    socket_path = Path(socket_path).resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(args)

    timeout = float(os.getenv("DEPTH_START_TIMEOUT", "5") or 5)

    async def wait_for_socket() -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(socket_path)
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"depth_service socket {socket_path} not available after {timeout}s"
                    )
                await asyncio.sleep(0.05)
            else:
                writer.close()
                await writer.wait_closed()
                return

    try:
        asyncio.run(wait_for_socket())
    except RuntimeError:
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=1)
        raise RuntimeError(
            f"Failed to start depth_service within {timeout}s"
        )
    return proc


async def depth_service_watchdog(
    cfg: dict, proc_ref: list[subprocess.Popen | None]
) -> None:
    """Monitor the ``depth_service`` process and attempt limited restarts."""
    proc = proc_ref[0]
    if not proc:
        return

    max_restarts = int(
        os.getenv("DEPTH_MAX_RESTARTS") or cfg.get("depth_max_restarts", 1)
    )
    restart_count = 0

    try:
        while True:
            await asyncio.sleep(1.0)
            if proc.poll() is None:
                continue
            if restart_count >= max_restarts:
                logging.error(
                    "depth_service exited after %d restarts; aborting",
                    restart_count,
                )
                raise RuntimeError("depth_service terminated")

            restart_count += 1
            logging.warning(
                "depth_service exited; attempting restart (%d/%d)",
                restart_count,
                max_restarts,
            )

            try:
                proc = await asyncio.to_thread(start_depth_service, cfg)
            except Exception as exc:
                logging.error("Failed to restart depth_service: %s", exc)
                raise
            if not proc:
                logging.error("depth_service restart returned None; aborting")
                raise RuntimeError("depth_service restart failed")
            proc_ref[0] = proc
    except asyncio.CancelledError:
        pass

