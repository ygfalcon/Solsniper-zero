from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from datetime import datetime
import tomllib

from .backtester import (
    backtest_strategies,
    backtest_configs,
    DEFAULT_STRATEGIES,
)


def _load_history(path: str, start: str | None, end: str | None) -> list[float]:
    """Load price history from ``path`` filtered by ``start`` and ``end`` dates."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return []

    if isinstance(data[0], dict) and "date" in data[0]:
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        filtered = []
        for item in data:
            d = datetime.fromisoformat(item["date"])
            if start_dt and d < start_dt:
                continue
            if end_dt and d > end_dt:
                continue
            filtered.append(float(item["price"]))
        return filtered

    if start or end:
        raise ValueError("Date range specified but history has no dates")

    return [float(x) for x in data]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Run strategy backtests")
    parser.add_argument("history", help="JSON file with price history list")
    parser.add_argument(
        "-c",
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="Configuration file with agent weights",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        dest="strategies",
        action="append",
        help="Strategy name to test",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args(argv)

    prices = _load_history(args.history, args.start, args.end)

    strategy_map = dict(DEFAULT_STRATEGIES)
    if args.strategies:
        strategies = []
        for name in args.strategies:
            if name not in strategy_map:
                parser.error(f"Unknown strategy: {name}")
            strategies.append((name, strategy_map[name]))
    else:
        strategies = DEFAULT_STRATEGIES

    if not args.configs:
        results = backtest_strategies(prices, strategies)
    else:
        cfgs = []
        for path in args.configs:
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
            weights = cfg.get("agent_weights", {})
            name = os.path.basename(path)
            cfgs.append((name, {str(k): float(v) for k, v in weights.items()}))
        results = backtest_configs(prices, cfgs, strategies)

    for res in results:
        print(f"{res.name}\tROI={res.roi:.4f}\tSharpe={res.sharpe:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
