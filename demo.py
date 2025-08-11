#!/usr/bin/env python3
"""Run the minimal trading demo with a configurable price source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.trading_demo import run_demo


def run(argv: list[str] | None = None) -> None:
    """Execute the trading demo.

    The demo expects a sequence of prices either provided via the ``--prices``
    argument or loaded from the built-in sample dataset.  The price list is
    forwarded to :func:`solhunter_zero.trading_demo.run_demo` which performs the
    simplified trading loop and writes the ROI report to the specified
    ``--reports`` directory.
    """

    parser = argparse.ArgumentParser(description="Run the trading demo")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory where reports will be written",
    )
    parser.add_argument(
        "--prices",
        type=str,
        default=None,
        help="Comma-separated list of prices to use instead of the sample dataset",
    )
    args = parser.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    if args.prices:
        prices = [float(p) for p in args.prices.split(",") if p]
    else:
        ticks = load_sample_ticks()
        prices = [t["price"] for t in ticks]

    run_demo(prices, args.reports)


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])
