"""Minimal paper trading CLI using :mod:`solhunter_zero` strategies.

This script loads a price history, runs the demo strategies from
``solhunter_zero.investor_demo`` and emits summary and trade history reports
similar to :func:`investor_demo.main`.  It serves as a lightweight example of
how the live bot evaluates strategies against historical data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict

from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.investor_demo import (
    DEFAULT_STRATEGIES,
    compute_weighted_returns,
    max_drawdown,
)


def _load_prices(path: Path | None) -> tuple[List[float], List[str]]:
    """Return ``(prices, dates)`` from a tick dataset."""

    ticks = load_sample_ticks(path) if path is not None else load_sample_ticks()
    if not ticks:
        raise ValueError("tick dataset is empty")
    prices: List[float] = []
    dates: List[str] = []
    for i, entry in enumerate(ticks):
        if "price" in entry:
            prices.append(float(entry["price"]))
            dates.append(str(entry.get("timestamp", i)))
    if not prices:
        raise ValueError("tick dataset missing 'price' entries")
    return prices, dates


def run(argv: List[str] | None = None) -> None:
    """Execute a tiny paper trading simulation."""

    parser = argparse.ArgumentParser(description="Run simple paper trading")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory to write summary and trade history reports",
    )
    parser.add_argument(
        "--ticks",
        type=Path,
        default=None,
        help="Path to JSON tick history (defaults to bundled sample)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100.0,
        help="Starting capital for the backtest",
    )
    args = parser.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    prices, dates = _load_prices(args.ticks)

    summary: List[Dict[str, float | int | str]] = []
    trade_history: List[Dict[str, float | int | str]] = []

    for name, _strat in DEFAULT_STRATEGIES:
        returns = compute_weighted_returns(prices, {name: 1.0})
        if returns:
            capital = args.capital
            cum: List[float] = []
            trades = wins = losses = 0
            for r in returns:
                capital *= 1 + r
                cum.append(capital / args.capital)
                if r != 0:
                    trades += 1
                    if r > 0:
                        wins += 1
                    else:
                        losses += 1
            roi = capital / args.capital - 1
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            vol = variance ** 0.5
            sharpe = mean / vol if vol else 0.0
            win_rate = wins / trades if trades else 0.0
        else:
            capital = args.capital
            roi = sharpe = vol = 0.0
            trades = wins = losses = 0
            win_rate = 0.0
            cum = []
        dd = max_drawdown(returns)
        metrics: Dict[str, float | int | str] = {
            "strategy": name,
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
            "volatility": vol,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "final_capital": capital,
        }
        summary.append(metrics)

        # Derive a trade log from the strategy returns
        capital = args.capital
        trade_history.append(
            {
                "strategy": name,
                "period": 0,
                "date": dates[0],
                "action": "buy",
                "price": prices[0],
                "capital": capital,
            }
        )
        for i in range(1, len(prices)):
            r = returns[i - 1] if i - 1 < len(returns) else 0.0
            capital *= 1 + r
            action = "buy" if r > 0 else "sell" if r < 0 else "hold"
            trade_history.append(
                {
                    "strategy": name,
                    "period": i,
                    "date": dates[i],
                    "action": action,
                    "price": prices[i],
                    "capital": capital,
                }
            )
        for i in range(len(returns) + 1, len(prices)):
            trade_history.append(
                {
                    "strategy": name,
                    "period": i,
                    "date": dates[i],
                    "action": "hold",
                    "price": prices[i],
                    "capital": capital,
                }
            )

    # Persist reports mirroring investor_demo.main
    with open(args.reports / "summary.json", "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    with open(args.reports / "trade_history.json", "w", encoding="utf-8") as hf:
        json.dump(trade_history, hf, indent=2)

    with open(args.reports / "summary.csv", "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=summary[0].keys())
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    with open(args.reports / "trade_history.csv", "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=trade_history[0].keys())
        writer.writeheader()
        for row in trade_history:
            writer.writerow(row)

    # Aggregated summary similar to investor_demo.main
    total_roi = sum(float(r["roi"]) for r in summary)
    total_sharpes = [float(r["sharpe"]) for r in summary]
    avg_sharpe = sum(total_sharpes) / len(total_sharpes) if total_sharpes else 0.0
    best_strategy = max(summary, key=lambda r: float(r["roi"]))["strategy"]
    worst_strategy = min(summary, key=lambda r: float(r["roi"]))["strategy"]
    aggregated = {
        "total_roi": total_roi,
        "average_sharpe": avg_sharpe,
        "best_strategy": best_strategy,
        "worst_strategy": worst_strategy,
    }
    with open(
        args.reports / "aggregated_summary.json", "w", encoding="utf-8"
    ) as agf:
        json.dump(aggregated, agf, indent=2)


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])


__all__ = ["run"]

