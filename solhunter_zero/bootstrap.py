from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict

from scripts.startup import (
    ensure_config,
    ensure_depth_service,
    ensure_deps,
    ensure_keypair,
    ensure_route_ffi,
    ensure_venv,
)

# Order of steps to report.  Both ``bootstrap`` and ``scripts/startup`` use
# this to produce a consistent structure other tools can rely on.
STEP_NAMES = [
    "venv",
    "deps",
    "config",
    "keypair",
    "route_ffi",
    "depth_service",
    "endpoints",
    "wallet_cli",
    "rpc",
    "cargo",
    "preflight",
]

REPORT_PATH = Path("startup_report.json")


def record_step(
    report: Dict[str, str],
    name: str,
    func: Callable[[], None],
    *,
    skip: bool = False,
) -> None:
    """Execute ``func`` and record ``name`` as success/failed/skipped."""

    if skip:
        report[name] = "skipped"
        return
    try:
        func()
    except BaseException:
        report[name] = "failed"
        raise
    else:
        report[name] = "success"


def finalize_report(report: Dict[str, str], path: Path = REPORT_PATH) -> None:
    """Print a text summary and persist ``report`` as JSON."""

    print("Check results:")
    for step in STEP_NAMES:
        status = report.get(step, "skipped")
        print(f"  {step}: {status}")
    try:
        path.write_text(json.dumps(report, indent=2))
    except Exception:
        pass
    else:
        print(f"Report written to {path}")


def bootstrap(
    one_click: bool = False, report: Dict[str, str] | None = None
) -> Dict[str, str]:
    """Initialize the runtime environment for SolHunter Zero.

    This helper mirrors the setup performed by ``scripts/startup.py`` and can
    be used by entry points that need to guarantee the project is ready to run
    programmatically.
    """

    own_report = report is None
    report = report or {step: "skipped" for step in STEP_NAMES}

    if one_click:
        os.environ.setdefault("AUTO_SELECT_KEYPAIR", "1")

    record_step(
        report,
        "venv",
        lambda: ensure_venv(None),
        skip=os.getenv("SOLHUNTER_SKIP_VENV") == "1",
    )

    record_step(
        report,
        "deps",
        lambda: ensure_deps(
            install_optional=os.getenv("SOLHUNTER_INSTALL_OPTIONAL") == "1"
        ),
        skip=os.getenv("SOLHUNTER_SKIP_DEPS") == "1",
    )

    if os.getenv("SOLHUNTER_SKIP_SETUP") == "1":
        report["config"] = "skipped"
        report["keypair"] = "skipped"
    else:
        record_step(report, "config", ensure_config)
        record_step(report, "keypair", ensure_keypair)

    record_step(report, "route_ffi", ensure_route_ffi)
    record_step(report, "depth_service", ensure_depth_service)

    if own_report:
        finalize_report(report)
    return report
