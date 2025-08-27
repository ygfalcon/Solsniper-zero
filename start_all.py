"""Runtime startup health gates for SolHunter Zero.

This lightweight entrypoint performs a series of health checks before the
full application is allowed to start.  The UI self-test is executed first and
any non-zero exit code aborts immediately.  A healthy RL daemon is now a
*mandatory* requirement – startup will fail fast if the RL service cannot be
reached.

The individual checks can be relaxed via environment variables which default
to paper-safe values so that tests can run without heavy services.
"""

from __future__ import annotations

from solhunter_zero.ui import ui_selftest
from solhunter_zero.health_runtime import (
    check_redis,
    check_depth_service,
    check_ffi,
    wait_for,
    http_ok,
)
import os
import logging


log = logging.getLogger("start_all")


def main() -> int:
    """Execute startup health gates.

    Returns ``0`` when all required checks pass.  ``SystemExit`` is raised with
    a descriptive message when any gate fails.
    """

    rc = ui_selftest()
    if rc != 0:
        raise SystemExit(rc)

    # Runtime health gates (paper-safe by default; can be relaxed via env)
    EVENT_BUS_URL = os.getenv("EVENT_BUS_URL", "redis://127.0.0.1:6379/0")
    # RL daemon is MANDATORY
    RL_HEALTH_URL = os.getenv("RL_HEALTH_URL", "http://127.0.0.1:7070/health")
    UI_HEALTH_URL = os.getenv("UI_HEALTH_URL", "http://127.0.0.1:3000/healthz")

    if os.getenv("USE_REDIS", "1") == "1":
        ok, msg = wait_for(lambda: check_redis(EVENT_BUS_URL))
        if not ok:
            raise SystemExit(f"Redis gate failed: {msg}")
        log.info("Redis reachable: %s", EVENT_BUS_URL)

    # RL daemon is required — do not proceed without a healthy RL service
    ok, msg = wait_for(lambda: http_ok(RL_HEALTH_URL), retries=60, sleep=1.0)
    if not ok:
        raise SystemExit(f"RL daemon gate failed: {msg}")
    log.info("RL daemon healthy: %s", RL_HEALTH_URL)

    if os.getenv("CHECK_UI_HEALTH", "1") == "1":
        ok, msg = wait_for(lambda: http_ok(UI_HEALTH_URL))
        if not ok:
            raise SystemExit(f"UI gate failed: {msg}")
        log.info("UI healthy: %s", UI_HEALTH_URL)

    # Allow CI to stop after gates so jobs don't hang:
    if os.getenv("START_ALL_EXIT_AFTER_GATES", "0") == "1":
        log.info("All gates green; exiting due to START_ALL_EXIT_AFTER_GATES=1")
        return 0

    if os.getenv("CHECK_DEPTH_SERVICE", "0") == "1":
        ok, msg = wait_for(check_depth_service)
        if not ok:
            raise SystemExit(f"Depth service gate failed: {msg}")
        log.info("Depth service ready")

    if os.getenv("CHECK_FFI", "0") == "1":
        ok, msg = wait_for(check_ffi)
        if not ok:
            raise SystemExit(f"FFI gate failed: {msg}")
        log.info("FFI ready")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

