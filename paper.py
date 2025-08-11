#!/usr/bin/env python3
"""Paper trading CLI producing ROI summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.simple_memory import SimpleMemory
from solhunter_zero.trade_analyzer import TradeAnalyzer


class SyncMemory(SimpleMemory):
    """Synchronous wrapper around :class:`SimpleMemory`."""

    def log_trade(self, **kwargs):  # type: ignore[override]
        return asyncio.run(super().log_trade(**kwargs))

    def list_trades(self, *args, **kwargs):  # type: ignore[override]
        trades = asyncio.run(super().list_trades(*args, **kwargs))
        return [SimpleNamespace(**t) for t in trades]


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
    mem = SyncMemory()
    first, last = ticks[0]["price"], ticks[-1]["price"]
    mem.log_trade(token="DEMO", direction="buy", amount=1.0, price=first, reason="demo")
    mem.log_trade(token="DEMO", direction="sell", amount=1.0, price=last, reason="demo")

    roi = TradeAnalyzer(mem).roi_by_agent()
    out_path = args.reports / "paper_roi.json"
    out_path.write_text(json.dumps(roi, indent=2))
    print("ROI by agent:", json.dumps(roi))
    print(f"Wrote paper trading report to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])

