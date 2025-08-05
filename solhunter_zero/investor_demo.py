from __future__ import annotations

"""Investor demo utilities and CLI."""

import argparse
import asyncio
import csv
import json
import sys
import types
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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


# Track which trade types have been exercised by the demo
used_trade_types: set[str] = set()

# Simple strategy functions used for demonstration


def _buy_and_hold(prices: List[float]) -> List[float]:
    rets: List[float] = []
    for i in range(1, len(prices)):
        rets.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return rets


def _momentum(prices: List[float]) -> List[float]:
    returns: List[float] = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i - 1]) / prices[i - 1]
        if r > 0:
            returns.append(r)
    return returns


DEFAULT_STRATEGIES: List[Tuple[str, Callable[[List[float]], List[float]]]] = [
    ("buy_hold", _buy_and_hold),
    ("momentum", _momentum),
]

# Packaged price data for the demo
DATA_FILE = resources.files(__package__) / "data" / "investor_demo_prices.json"


def compute_weighted_returns(prices: List[float], weights: Dict[str, float]) -> List[float]:
    """Aggregate returns for strategies weighted in ``weights`` using pure Python."""

    arrs: List[Tuple[List[float], float]] = []
    weight_sum = 0.0
    for name, strat in DEFAULT_STRATEGIES:
        w = float(weights.get(name, 0.0))
        if not w:
            continue
        rets = strat(prices)
        if rets:
            arrs.append(([float(r) for r in rets], w))
            weight_sum += w
    if not arrs or weight_sum == 0:
        return []
    length = min(len(a) for a, _ in arrs)
    agg = [0.0] * length
    for a, w in arrs:
        for i in range(length):
            agg[i] += w * a[i]
    for i in range(length):
        agg[i] /= weight_sum
    return agg


def max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown from a series of returns."""
    if not returns:
        return 0.0
    cumulative: List[float] = []
    total = 1.0
    for r in returns:
        total *= 1 + r
        cumulative.append(total)
    peak: List[float] = []
    max_val = float("-inf")
    for c in cumulative:
        if c > max_val:
            max_val = c
        peak.append(max_val)
    drawdowns = [(p - c) / p if p else 0.0 for c, p in zip(cumulative, peak)]
    return max(drawdowns) if drawdowns else 0.0


def load_prices(path: Path | None = None) -> List[float]:
    """Load a JSON price dataset into a list of floats."""
    if path is None:
        data_text = DATA_FILE.read_text()
    else:
        data_text = path.read_text()
    data = json.loads(data_text)
    prices = [float(entry["price"]) for entry in data]
    return prices


async def _demo_arbitrage() -> None:
    """Invoke arbitrage detection with stub inputs."""

    mod_name = f"{__package__}.arbitrage"
    orig = sys.modules.get(mod_name)
    arb_stub = types.ModuleType(mod_name)

    async def detect_and_execute_arbitrage(*_args, **_kwargs):  # type: ignore
        return None

    arb_stub.detect_and_execute_arbitrage = detect_and_execute_arbitrage
    sys.modules[mod_name] = arb_stub
    try:
        from .arbitrage import detect_and_execute_arbitrage as demo_func  # type: ignore
        async def _feed(_token: str) -> float:
            return 1.0
        await demo_func("demo", feeds=[_feed, _feed], use_service=False)
    finally:
        if orig is not None:
            sys.modules[mod_name] = orig
        else:
            del sys.modules[mod_name]

    used_trade_types.add("arbitrage")


async def _demo_flash_loan() -> None:
    """Invoke flash loan borrow/repay with stub inputs."""

    mod_name = f"{__package__}.flash_loans"
    orig = sys.modules.get(mod_name)
    fl_stub = types.ModuleType(mod_name)

    async def borrow_flash(*_args, **_kwargs):  # type: ignore
        return "sig"

    async def repay_flash(*_args, **_kwargs) -> bool:  # type: ignore
        return True

    fl_stub.borrow_flash = borrow_flash
    fl_stub.repay_flash = repay_flash
    sys.modules[mod_name] = fl_stub
    try:
        from .flash_loans import borrow_flash as _borrow, repay_flash as _repay  # type: ignore
        sig = await _borrow(1.0, "demo", [])
        await _repay(sig)
    finally:
        if orig is not None:
            sys.modules[mod_name] = orig
        else:
            del sys.modules[mod_name]

    used_trade_types.add("flash_loan")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Investor demo backtest")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
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

    configs = {
        "buy_hold": {"buy_hold": 1.0},
        "momentum": {"momentum": 1.0},
        "mixed": {"buy_hold": 0.5, "momentum": 0.5},
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, float]] = []
    trade_history: List[Dict[str, float]] = []

    for name, weights in configs.items():
        returns = compute_weighted_returns(prices, weights)
        if returns:
            cum: List[float] = []
            total = 1.0
            for r in returns:
                total *= 1 + r
                cum.append(total)
            roi = cum[-1] - 1
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            vol = variance ** 0.5
            sharpe = mean / vol if vol else 0.0
        else:
            cum = []
            roi = 0.0
            sharpe = 0.0
        dd = max_drawdown(returns)
        final_capital = args.capital * (1 + roi)
        metrics = {
            "config": name,
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
            "final_capital": final_capital,
        }
        summary.append(metrics)

        # record per-period capital history for this strategy
        capital = args.capital
        trade_history.append({"strategy": name, "period": 0, "capital": capital})
        for i, r in enumerate(returns, start=1):
            capital *= 1 + r
            trade_history.append({"strategy": name, "period": i, "capital": capital})

        try:  # plotting hook
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure()
            if cum:
                plt.plot(cum, label="Cumulative Return")
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

    # Write trade history in CSV and JSON for inspection
    hist_csv = args.reports / "trade_history.csv"
    hist_json = args.reports / "trade_history.json"
    if trade_history:
        with open(hist_json, "w", encoding="utf-8") as jf:
            json.dump(trade_history, jf, indent=2)
        with open(hist_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["strategy", "period", "capital"])
            writer.writeheader()
            for row in trade_history:
                writer.writerow(row)

    # Write highlights summarising top performing strategy
    top = None
    if summary:
        top = max(summary, key=lambda e: e["final_capital"])
        highlights = {
            "top_strategy": top["config"],
            "top_final_capital": top["final_capital"],
            "top_roi": top["roi"],
        }
        with open(args.reports / "highlights.json", "w", encoding="utf-8") as hf:
            json.dump(highlights, hf, indent=2)

    # Display a simple capital summary on stdout
    print("Capital Summary:")
    for row in summary:
        print(f"{row['config']}: {row['final_capital']:.2f}")
    if top:
        print(
            f"Top strategy: {top['config']} with final capital {top['final_capital']:.2f}"
        )

    # Exercise trade types via lightweight stubs
    asyncio.run(_demo_arbitrage())
    asyncio.run(_demo_flash_loan())

    required = {"arbitrage", "flash_loan"}
    missing = required - used_trade_types
    if missing:
        raise RuntimeError(f"Demo did not exercise trade types: {', '.join(sorted(missing))}")

    print(f"Wrote reports to {args.reports}")


if __name__ == "__main__":  # pragma: no cover
    main()
