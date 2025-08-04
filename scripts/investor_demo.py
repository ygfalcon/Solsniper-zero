#!/usr/bin/env python
"""Demonstration script for investors running rolling backtests.

This script runs a rolling backtest over a historical price dataset using
several strategy weight configurations. It outputs a JSON and CSV report with
ROI, Sharpe ratio and drawdown metrics. Optional performance plots are saved to
``reports``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure repository root is on path when executed as a script
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a lightweight Memory stub to avoid heavy dependencies in demos
import types
memory_stub = types.ModuleType("solhunter_zero.memory")

class Memory:  # minimal stub
    def __init__(self, *args, **kwargs):
        pass

memory_stub.Memory = Memory
sys.modules.setdefault("solhunter_zero.memory", memory_stub)

# Stub portfolio module to avoid optional dependencies
portfolio_stub = types.ModuleType("solhunter_zero.portfolio")

def hedge_allocation(*args, **kwargs):
    return 0.0

portfolio_stub.hedge_allocation = hedge_allocation
sys.modules.setdefault("solhunter_zero.portfolio", portfolio_stub)

from solhunter_zero.backtest_pipeline import rolling_backtest
from solhunter_zero.backtester import DEFAULT_STRATEGIES
from solhunter_zero.risk import RiskManager


def compute_weighted_returns(prices: List[float], weights: Dict[str, float]) -> np.ndarray:
    """Return aggregated strategy returns for ``weights``."""
    weight_sum = sum(float(weights.get(name, 1.0)) for name, _ in DEFAULT_STRATEGIES)
    arrs = []
    for name, strat in DEFAULT_STRATEGIES:
        rets = strat(prices)
        if rets:
            arrs.append((np.array(rets, dtype=float), float(weights.get(name, 1.0))))
    if not arrs or weight_sum == 0:
        return np.array([])
    length = min(len(a) for a, _ in arrs)
    agg = np.zeros(length, dtype=float)
    for a, w in arrs:
        agg += w * a[:length]
    agg /= weight_sum
    return agg


def max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from a series of returns."""
    if returns.size == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak
    return float(np.max(drawdowns))


def load_prices(path: Path) -> List[float]:
    """Load a JSON price dataset into a list of floats."""
    data = json.loads(path.read_text())
    prices = [float(entry["price"]) for entry in data]
    return prices


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Investor demo backtest")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("tests/data/prices.json"),
        help="Path to JSON price history",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory to write reports/plots to",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100.0,
        help="Starting capital for the backtest",
    )
    args = parser.parse_args(argv)

    prices = load_prices(args.data)
    history = {"asset": prices}

    configs = {
        "buy_hold": {"buy_hold": 1.0},
        "momentum": {"momentum": 1.0},
        "mixed": {"buy_hold": 0.5, "momentum": 0.5},
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, float]] = []

    for name, weights in configs.items():
        risk = RiskManager(min_portfolio_value=args.capital)
        result = rolling_backtest(history, weights, risk)
        returns = compute_weighted_returns(prices, weights)
        dd = max_drawdown(returns)
        final_capital = args.capital * (1 + result.roi)
        metrics = {
            "config": name,
            "roi": result.roi,
            "sharpe": result.sharpe,
            "drawdown": dd,
            "final_capital": final_capital,
        }
        summary.append(metrics)

        try:  # plotting hook
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure()
            if returns.size:
                plt.plot(np.cumprod(1 + returns), label="Cumulative Return")
            plt.title(f"Performance - {name}")
            plt.xlabel("Period")
            plt.ylabel("Growth")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.reports / f"{name}.png")
            plt.close()
        except Exception as exc:  # pragma: no cover - plotting optional
            print(
                f"Plotting failed for {name}: {exc}. "
                "Install the demo extra with 'pip install solhunter-zero[demo]' to enable plotting."
            )

    json_path = args.reports / "summary.json"
    csv_path = args.reports / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=summary[0].keys())
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"Wrote reports to {args.reports}")


if __name__ == "__main__":
    main()
