from __future__ import annotations

import json
from argparse import ArgumentParser

from .backtester import backtest_strategies


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run strategy backtests")
    parser.add_argument("history", help="JSON file with price history list")
    args = parser.parse_args(argv)

    with open(args.history, "r", encoding="utf-8") as f:
        prices = json.load(f)

    results = backtest_strategies(prices)
    for res in results:
        print(f"{res.name}\tROI={res.roi:.4f}\tSharpe={res.sharpe:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
