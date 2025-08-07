#!/usr/bin/env python3
"""Run Solsniper Zero with minimal user interaction."""

from __future__ import annotations

import platform
import sys

from solhunter_zero.bootstrap_utils import ensure_venv


def main(argv: list[str] | None = None) -> None:
    """Execute preflight checks then launch the existing startup routine."""
    ensure_venv(None)
    if platform.system() == "Darwin":
        from scripts import mac_env
        mac_env.prepare_macos_env()
    from scripts import self_test, launcher
    results = self_test.run_self_test()
    print("Self-test summary:")
    for item in results["preflight"]:
        status = "OK" if item["ok"] else "FAIL"
        print(f"{item['name']}: {status} - {item['message']}")
    gpu = results["gpu"]
    gpu_status = "OK" if gpu["ok"] else "FAIL"
    print(f"GPU Verification: {gpu_status} - {gpu['message']}")
    net = results["network"]
    net_status = "OK" if net["ok"] else "FAIL"
    print(f"Network: {net_status} - {net['message']}")
    if not (
        all(item["ok"] for item in results["preflight"])
        and gpu["ok"]
        and net["ok"]
    ):
        sys.exit(1)
    launcher.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
