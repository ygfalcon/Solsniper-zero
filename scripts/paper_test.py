from __future__ import annotations

import argparse
import os
import asyncio

from solhunter_zero.main import run_auto
from solhunter_zero.memory import Memory
from solhunter_zero.trade_analyzer import TradeAnalyzer
from solhunter_zero.util import run_coro
from solhunter_zero.http import close_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a dry-run trading loop")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of loop iterations to execute",
    )
    parser.add_argument(
        "--memory",
        default="sqlite:///memory.db",
        help="Memory database URL",
    )
    args = parser.parse_args(argv)

    if args.config:
        os.environ["SOLHUNTER_CONFIG"] = args.config

    run_auto(
        memory_path=args.memory,
        iterations=args.iterations,
        dry_run=True,
        offline=True,
    )

    mem = Memory(args.memory)
    analyzer = TradeAnalyzer(mem)
    roi_by_agent = analyzer.roi_by_agent()
    print("ROI by agent:", roi_by_agent)

    trades = run_coro(mem.list_trades(limit=1000))
    spent = sum(float(t.amount) * float(t.price) for t in trades if t.direction == "buy")
    revenue = sum(float(t.amount) * float(t.price) for t in trades if t.direction == "sell")
    roi = (revenue - spent) / spent if spent > 0 else 0.0
    print(f"Overall ROI: {roi:.4f}")

    run_coro(mem.close())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        asyncio.run(close_session())
