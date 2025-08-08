#!/usr/bin/env python3
"""Runtime startup verification helpers."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Tuple

from solhunter_zero.paths import ROOT
from solhunter_zero import event_bus
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.config import apply_env_overrides, load_config


def _check_preflight_log() -> Tuple[bool, str]:
    """Ensure the preflight log exists and reports success."""
    log_path = ROOT / "preflight.log"
    if not log_path.exists():
        return False, "preflight.log not found"
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - unexpected I/O
        return False, f"failed reading preflight.log: {exc}"
    if "FAIL" in text or "error" in text.lower():
        return False, "preflight log contains failures"
    return True, "preflight log OK"


async def _wait_heartbeat(timeout: float = 5.0) -> Tuple[bool, str]:
    """Wait for a heartbeat event from the trading loop."""
    event = asyncio.Event()

    def _handler(_payload: object) -> None:
        event.set()

    with event_bus.subscription("heartbeat", _handler):
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            return False, "no heartbeat received"
    return True, "heartbeat received"


def _check_agents() -> Tuple[bool, str]:
    """Verify at least one agent is registered via AgentManager."""
    try:
        cfg = apply_env_overrides(load_config())
        mgr = AgentManager.from_config(cfg)
    except Exception as exc:  # pragma: no cover - config issues
        return False, f"agent manager init failed: {exc}"
    if not mgr or not getattr(mgr, "agents", []):
        return False, "no agents registered"
    mgr.close()
    return True, "agents registered"


def _check_ui(timeout: float = 5.0) -> Tuple[bool, str]:
    """Probe UI HTTP endpoint for availability."""
    import urllib.request

    port = int(os.getenv("UI_PORT") or os.getenv("PORT", "5000"))
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:  # nosec B310
                if resp.status < 400:
                    return True, f"UI reachable on port {port}"
        except Exception:
            time.sleep(0.5)
            continue
        return False, f"UI responded with status {resp.status}"
    return False, f"UI not reachable on port {port}"


def main(timeout: float = 5.0) -> bool:
    """Run all startup verification checks."""
    results = []
    results.append(("preflight", *_check_preflight_log()))
    results.append(("heartbeat", *asyncio.run(_wait_heartbeat(timeout))))
    results.append(("agents", *_check_agents()))
    results.append(("ui", *_check_ui(timeout)))

    ok = True
    for name, success, msg in results:
        if not success:
            ok = False
            print(f"{name} check failed: {msg}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if main() else 1)
