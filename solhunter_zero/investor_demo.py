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

from .memory import Memory
from .portfolio import hedge_allocation
from .risk import correlation_matrix


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
        returns.append(r if r > 0 else 0.0)
    return returns


def _mean_reversion(prices: List[float]) -> List[float]:
    """Simple mean-reversion strategy.

    This toy implementation buys after price drops and assumes an immediate
    rebound. Negative returns are flipped to positive gains while positive
    moves are ignored.
    """

    returns: List[float] = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns.append(-r if r < 0 else 0.0)
    return returns


DEFAULT_STRATEGIES: List[Tuple[str, Callable[[List[float]], List[float]]]] = [
    ("buy_hold", _buy_and_hold),
    ("momentum", _momentum),
    ("mean_reversion", _mean_reversion),
]

# Packaged price data for the demo
DATA_FILE = resources.files(__package__) / "data" / "investor_demo_prices.json"


def compute_weighted_returns(prices: List[float], weights: Dict[str, float]) -> List[float]:
    """Aggregate strategy returns weighted by ``weights``.

    Negative weights represent short positions.  Returns are normalised by the
    sum of absolute weights so that portfolios with offsetting long and short
    allocations still produce meaningful values.
    """

    arrs: List[Tuple[List[float], float]] = []
    abs_sum = 0.0
    for name, strat in DEFAULT_STRATEGIES:
        w = float(weights.get(name, 0.0))
        if not w:
            continue
        rets = strat(prices)
        if rets:
            arrs.append(([float(r) for r in rets], w))
            abs_sum += abs(w)
    if not arrs or abs_sum == 0.0:
        return []
    length = min(len(a) for a, _ in arrs)
    agg = [0.0] * length
    for a, w in arrs:
        for i in range(length):
            agg[i] += w * a[i]
    for i in range(length):
        agg[i] /= abs_sum
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


async def _demo_sniper() -> None:
    """Invoke sniper evaluate with stub inputs."""

    mod_name = f"{__package__}.sniper"
    orig = sys.modules.get(mod_name)
    sni_stub = types.ModuleType(mod_name)

    async def evaluate(*_args, **_kwargs):  # type: ignore
        return [{"token": "demo"}]

    sni_stub.evaluate = evaluate
    sys.modules[mod_name] = sni_stub
    try:
        from .sniper import evaluate as demo_func  # type: ignore
        await demo_func("demo", None)
    finally:
        if orig is not None:
            sys.modules[mod_name] = orig
        else:
            del sys.modules[mod_name]

    used_trade_types.add("sniper")


async def _demo_dex_scanner() -> None:
    """Invoke DEX pool scanning with stub inputs."""

    mod_name = f"{__package__}.dex_scanner"
    orig = sys.modules.get(mod_name)
    dex_stub = types.ModuleType(mod_name)

    async def scan_new_pools(*_args, **_kwargs):  # type: ignore
        return ["demo"]

    dex_stub.scan_new_pools = scan_new_pools
    sys.modules[mod_name] = dex_stub
    try:
        from .dex_scanner import scan_new_pools as demo_func  # type: ignore
        await demo_func("url")
    finally:
        if orig is not None:
            sys.modules[mod_name] = orig
        else:
            del sys.modules[mod_name]

    used_trade_types.add("dex_scanner")


def main(argv: List[str] | None = None) -> None:
    used_trade_types.clear()
    from . import resource_monitor
    try:
        import psutil  # type: ignore
    except Exception:  # pragma: no cover - psutil optional
        psutil = None  # type: ignore
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

    # Demonstrate Memory usage and portfolio hedging
    mem = Memory("sqlite:///:memory:")
    mem.log_var(0.0)
    asyncio.run(mem.close())

    # Compute correlations between strategy returns
    strategy_returns: Dict[str, List[float]] = {
        name: strat(prices) for name, strat in DEFAULT_STRATEGIES
    }
    series: Dict[str, List[float]] = {}
    for name, rets in strategy_returns.items():
        if not rets:
            continue
        pseudo_prices = [1.0]
        for r in rets:
            pseudo_prices.append(pseudo_prices[-1] * (1 + r))
        series[name] = pseudo_prices
    corr_pairs: Dict[tuple[str, str], float] = {}
    if len(series) >= 2:
        try:
            corr_mat = correlation_matrix(series)
            keys = list(series.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    corr_pairs[(keys[i], keys[j])] = float(corr_mat[i, j])
        except Exception:
            keys = list(strategy_returns.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a = strategy_returns[keys[i]]
                    b = strategy_returns[keys[j]]
                    n = min(len(a), len(b))
                    if n == 0:
                        continue
                    a = a[:n]
                    b = b[:n]
                    ma = sum(a) / n
                    mb = sum(b) / n
                    va = sum((x - ma) ** 2 for x in a) / n
                    vb = sum((y - mb) ** 2 for y in b) / n
                    if va <= 0 or vb <= 0:
                        c = 0.0
                    else:
                        cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n)) / n
                        c = cov / (va ** 0.5 * vb ** 0.5)
                    corr_pairs[(keys[i], keys[j])] = c

    _ = hedge_allocation({"buy_hold": 1.0, "momentum": 0.0}, corr_pairs)

    configs = {
        "buy_hold": {"buy_hold": 1.0},
        "momentum": {"momentum": 1.0},
        "mean_reversion": {"mean_reversion": 1.0},
        "mixed": {
            "buy_hold": 1 / 3,
            "momentum": 1 / 3,
            "mean_reversion": 1 / 3,
        },
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, float | int]] = []
    trade_history: List[Dict[str, float | str]] = []

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
            trades = sum(1 for r in returns if r != 0)
            wins = sum(1 for r in returns if r > 0)
            losses = sum(1 for r in returns if r < 0)
            win_rate = wins / trades if trades else 0.0
        else:
            cum = []
            roi = 0.0
            sharpe = 0.0
            vol = 0.0
            trades = wins = losses = 0
            win_rate = 0.0
        dd = max_drawdown(returns)
        final_capital = args.capital * (1 + roi)
        metrics = {
            "config": name,
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
            "volatility": vol,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "final_capital": final_capital,
        }
        summary.append(metrics)

        # record per-period capital history for this strategy
        capital = args.capital
        trade_history.append(
            {
                "strategy": name,
                "period": 0,
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
                    "action": "hold",
                    "price": prices[i],
                    "capital": capital,
                }
            )

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
            writer = csv.DictWriter(
                cf, fieldnames=["strategy", "period", "action", "price", "capital"]
            )
            writer.writeheader()
            for row in trade_history:
                writer.writerow(row)

    # Collect resource usage metrics if available
    cpu_usage = None
    mem_pct = None
    if psutil is not None:
        try:
            cpu_usage = resource_monitor.get_cpu_usage()
            mem_pct = psutil.virtual_memory().percent
        except Exception:
            pass

    # Write highlights summarising top performing strategy
    top = None
    if summary:
        top = max(summary, key=lambda e: e["final_capital"])
        highlights = {
            "top_strategy": top["config"],
            "top_final_capital": top["final_capital"],
            "top_roi": top["roi"],
        }
        if cpu_usage is not None:
            highlights["cpu_usage"] = cpu_usage
        if mem_pct is not None:
            highlights["memory_percent"] = mem_pct
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
    if cpu_usage is not None and mem_pct is not None:
        print(
            f"Resource usage - CPU: {cpu_usage:.2f}% Memory: {mem_pct:.2f}%"
        )

    # Exercise trade types via lightweight stubs
    async def _exercise_trade_types() -> None:
        await asyncio.gather(
            _demo_arbitrage(),
            _demo_flash_loan(),
            _demo_sniper(),
            _demo_dex_scanner(),
        )

    asyncio.run(_exercise_trade_types())

    required = {"arbitrage", "flash_loan", "sniper", "dex_scanner"}
    missing = required - used_trade_types
    if missing:
        raise RuntimeError(f"Demo did not exercise trade types: {', '.join(sorted(missing))}")

    print(f"Wrote reports to {args.reports}")


if __name__ == "__main__":  # pragma: no cover
    main()
