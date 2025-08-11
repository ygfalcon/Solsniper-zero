"""Paper trading CLI producing ROI summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.trading_demo import run_demo, SyncMemory


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a minimal paper trading demo")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory to write ROI summaries",
    )
    args = parser.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    ticks = load_sample_ticks()
    prices = [t["price"] for t in ticks]
    mem = SyncMemory()
    run_demo(prices, args.reports, memory=mem)


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])
