#!/usr/bin/env python3
"""Wrapper script for the one-click trading demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from solhunter_zero.investor_demo import run_one_click_trading_demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run demo strategy backtests")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory to write summary reports",
    )
    args = parser.parse_args(argv)
    run_one_click_trading_demo(args.reports)


if __name__ == "__main__":  # pragma: no cover
    main()
