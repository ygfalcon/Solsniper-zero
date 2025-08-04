from __future__ import annotations

"""Investor demo utilities and CLI."""

import argparse
import csv
import json
import sys
import types
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Provide lightweight stubs for optional heavy modules
memory_stub = types.ModuleType("solhunter_zero.memory")


class Memory:  # minimal stub
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        pass


def hedge_allocation(*args, **kwargs) -> float:  # pragma: no cover - simple stub
    return 0.0


memory_stub.Memory = Memory
sys.modules.setdefault("solhunter_zero.memory", memory_stub)

portfolio_stub = types.ModuleType("solhunter_zero.portfolio")
portfolio_stub.hedge_allocation = hedge_allocation
sys.modules.setdefault("solhunter_zero.portfolio", portfolio_stub)

from .backtest_pipeline import rolling_backtest
from .backtester import DEFAULT_STRATEGIES
from .risk import RiskManager

# Packaged price data for the demo
DATA_FILE = resources.files(__package__) / "data" / "investor_demo_prices.json"
DEFAULT_DATA_PATH = Path(DATA_FILE)


def compute_weighted_returns(prices: List[float], weights: Dict[str, float]) -> np.ndarray:
    """Aggregate returns only for strategies explicitly weighted in ``weights``.

    Strategies not present in ``weights`` or given a weight of zero are
    ignored when computing the combined return series.
    """

    arrs: List[Tuple[np.ndarray, float]] = []
    weight_sum = 0.0
    for name, strat in DEFAULT_STRATEGIES:
        w = float(weights.get(name, 0.0))
        if not w:
            continue
        rets = strat(prices)
        if rets:
            arrs.append((np.array(rets, dtype=float), w))
            weight_sum += w
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
        default=DEFAULT_DATA_PATH,
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
            print(f"Plotting failed for {name}: {exc}")

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


if __name__ == "__main__":  # pragma: no cover
    main()
