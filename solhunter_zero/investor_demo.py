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
from typing import Callable, Dict, List, Tuple, Union

try:  # SQLAlchemy is optional; fall back to a simple in-memory implementation
    from .memory import Memory  # type: ignore
except Exception:  # pragma: no cover - absence of SQLAlchemy
    Memory = None  # type: ignore[assignment]

from .portfolio import hedge_allocation
from .risk import correlation_matrix


# Track which trade types have been exercised by the demo
used_trade_types: set[str] = set()

# Toggle for heavier demo features such as real RL training
FULL_SYSTEM: bool = False

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

# Optional preset datasets bundled with the package
PRESET_DATA_FILES: Dict[str, Path] = {
    "short": resources.files(__package__) / "data" / "investor_demo_prices_short.json",
    "multi": resources.files(__package__) / "data" / "investor_demo_prices_multi.json",
}


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


def load_prices(
    path: Path | None = None,
    preset: str | None = None,
) -> Union[Tuple[List[float], List[str]], Dict[str, Tuple[List[float], List[str]]]]:
    """Load a JSON price dataset.

    The legacy dataset format is a list of ``{"date", "price"}`` objects which
    represents a single token.  To support scenarios with multiple tokens this
    function also accepts a mapping of token name to such a list.  When a
    mapping is provided the return value is a dictionary keyed by token whose
    values are ``(prices, dates)`` tuples.  For backwards compatibility a list
    input continues to return a single ``(prices, dates)`` tuple.
    """

    if path is not None and preset is not None:
        raise ValueError("Provide only one of 'path' or 'preset'")

    if preset is not None:
        data_path = PRESET_DATA_FILES.get(preset)
        if data_path is None:
            raise ValueError(f"Unknown preset '{preset}'")
        data_text = data_path.read_text()
    elif path is not None:
        data_text = path.read_text()
    else:
        data_text = DATA_FILE.read_text()
    data = json.loads(data_text)

    def _parse(entries: List[object]) -> Tuple[List[float], List[str]]:
        prices: List[float] = []
        dates: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or "price" not in entry or "date" not in entry:
                raise ValueError("Each entry must contain 'date' and numeric 'price'")
            price = entry["price"]
            date = entry["date"]
            if not isinstance(price, (int, float)):
                raise ValueError("Each entry must contain a numeric 'price'")
            if not isinstance(date, str):
                raise ValueError("Each entry must contain a string 'date'")
            prices.append(float(price))
            dates.append(date)
        return prices, dates

    if isinstance(data, list):
        return _parse(data)
    if isinstance(data, dict):
        out: Dict[str, Tuple[List[float], List[str]]] = {}
        for token, entries in data.items():
            if not isinstance(token, str) or not isinstance(entries, list):
                raise ValueError("Price data mapping must be token -> list of entries")
            out[token] = _parse(entries)
        return out
    raise ValueError("Price data must be a list or mapping of token to list")


async def _demo_arbitrage() -> Dict[str, object]:
    """Exercise the real :mod:`arbitrage` module with static prices."""

    from . import arbitrage

    prices = {"dex1": 100.0, "dex2": 105.0}
    fees = {"dex1": 0.0, "dex2": 0.0}
    gas = {"dex1": 0.0, "dex2": 0.0}
    latency = {"dex1": 0.0, "dex2": 0.0}
    depth = {
        "dex1": {"bids": 1_000.0, "asks": 1_000.0},
        "dex2": {"bids": 1_000.0, "asks": 1_000.0},
    }
    path, profit = arbitrage._best_route(  # type: ignore[attr-defined]
        prices,
        1.0,
        fees=fees,
        gas=gas,
        latency=latency,
        depth=depth,
        use_flash_loans=False,
        max_flash_amount=0.0,
        max_hops=2,
        use_gnn_routing=False,
    )
    used_trade_types.add("arbitrage")
    return {"path": path, "profit": float(profit)}


async def _demo_flash_loan() -> str | None:
    """Invoke :mod:`flash_loans.borrow_flash` with stubbed network calls."""

    from . import flash_loans, depth_client
    from solders.keypair import Keypair
    from solders.instruction import Instruction, AccountMeta
    from solders.pubkey import Pubkey
    from solders.hash import Hash
    import types

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_latest_blockhash(self):
            return types.SimpleNamespace(
                value=types.SimpleNamespace(blockhash=Hash.default())
            )

        async def confirm_transaction(self, _sig: str) -> None:
            return None

    async def dummy_submit(_tx: str) -> str:
        return "demo_sig"

    orig_client = flash_loans.AsyncClient
    orig_submit = depth_client.submit_raw_tx
    flash_loans.AsyncClient = lambda _url: DummyClient()  # type: ignore[assignment]
    depth_client.submit_raw_tx = dummy_submit  # type: ignore[assignment]
    try:
        payer = Keypair()
        ix = Instruction(
            Pubkey.default(), b"demo", [AccountMeta(payer.pubkey(), True, True)]
        )
        sig = await flash_loans.borrow_flash(
            1.0,
            "USDC",
            [ix],
            payer=payer,
            program_accounts={},
            rpc_url="http://offline",
        )
    finally:
        flash_loans.AsyncClient = orig_client  # type: ignore[assignment]
        depth_client.submit_raw_tx = orig_submit  # type: ignore[assignment]
    used_trade_types.add("flash_loan")
    return sig


async def _demo_sniper() -> List[str]:
    """Run :mod:`sniper.evaluate` on deterministic inputs."""

    import sys
    import types
    from .decision import should_buy, should_sell
    from .simulation import SimulationResult

    mem_mod = types.ModuleType("solhunter_zero.memory")
    mem_mod.Memory = type("Memory", (), {})  # minimal stub
    sys.modules.setdefault("solhunter_zero.memory", mem_mod)

    main_mod = types.ModuleType("solhunter_zero.main")

    async def fetch_prices(tokens: set[str]) -> Dict[str, float]:
        return {t: 1.0 for t in tokens}

    def run_sims(_token: str, count: int = 100) -> List[SimulationResult]:
        return [
            SimulationResult(
                success_prob=0.9,
                expected_roi=1.2,
                volume=1.0,
                liquidity=1.0,
                slippage=0.1,
                volatility=0.1,
                volume_spike=1.1,
                depth_change=0.1,
                whale_activity=0.1,
                tx_rate=1.0,
            ),
            SimulationResult(
                success_prob=0.8,
                expected_roi=1.1,
                volume=1.0,
                liquidity=1.0,
                slippage=0.1,
                volatility=0.1,
                volume_spike=1.1,
                depth_change=0.1,
                whale_activity=0.1,
                tx_rate=1.0,
            ),
        ]

    main_mod.run_simulations = run_sims  # type: ignore[attr-defined]
    main_mod.should_buy = should_buy  # type: ignore[attr-defined]
    main_mod.should_sell = should_sell  # type: ignore[attr-defined]
    main_mod.fetch_token_prices_async = fetch_prices  # type: ignore[attr-defined]
    sys.modules.setdefault("solhunter_zero.main", main_mod)

    from . import sniper
    from .portfolio import Portfolio, Position

    orig_predict = sniper.predict_price_movement
    sniper.predict_price_movement = lambda _t: 0.1
    try:
        port = Portfolio(path=None)
        port.balances["USD"] = Position("USD", 100.0, 1.0, 1.0)
        actions = await sniper.evaluate("TKN", port)
    finally:
        sniper.predict_price_movement = orig_predict
    used_trade_types.add("sniper")
    return [a["token"] for a in actions]


async def _demo_dex_scanner() -> List[str]:
    """Scan for new pools using :mod:`dex_scanner` with stubbed RPC."""

    from . import dex_scanner

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get_program_accounts(self, *_a, **_k):
            return {
                "result": [
                    {
                        "account": {
                            "data": {
                                "parsed": {
                                    "info": {
                                        "tokenA": {
                                            "mint": "mintA",
                                            "name": "AlphaBonk",
                                        },
                                        "tokenB": {
                                            "mint": "mintB",
                                            "name": "BetaBonk",
                                        },
                                    }
                                }
                            }
                        }
                    }
                ]
            }

    orig_client = dex_scanner.AsyncClient
    dex_scanner.AsyncClient = lambda _url: DummyClient()  # type: ignore[assignment]
    try:
        tokens = await dex_scanner.scan_new_pools("http://offline")
    finally:
        dex_scanner.AsyncClient = orig_client  # type: ignore[assignment]
    used_trade_types.add("dex_scanner")
    return tokens


def _demo_rl_agent() -> float:
    """Train a tiny RL model using the real pipeline and return reward.

    When :data:`FULL_SYSTEM` is ``False`` this function exits immediately so
    the demo remains lightweight.  When enabled it invokes the genuine
    reinforcement learning utilities to fit a minimal PPO model on a static
    dataset and reports the resulting reward for correctly predicted actions.
    """

    if not FULL_SYSTEM:
        return 0.0

    try:
        from datetime import datetime
        from types import SimpleNamespace
        import tempfile
        from pathlib import Path

        import torch  # type: ignore[import-untyped]
        from torch.utils.data import DataLoader  # type: ignore[import-untyped]

        from . import rl_training, simulation
        from .rl_training import _TradeDataset, LightningPPO
    except Exception:  # pragma: no cover - optional deps
        return 0.0

    # Avoid network calls for price prediction during dataset construction
    orig_predict = simulation.predict_price_movement
    simulation.predict_price_movement = lambda *a, **k: 0.0  # type: ignore
    try:
        now = datetime.utcnow()
        trades = [
            SimpleNamespace(
                token="demo",
                side="buy",
                price=1.0,
                amount=1.0,
                timestamp=now,
            ),
            SimpleNamespace(
                token="demo",
                side="sell",
                price=1.1,
                amount=1.0,
                timestamp=now,
            ),
        ]
        snaps = [
            SimpleNamespace(
                token="demo",
                depth=1.0,
                slippage=0.0,
                imbalance=0.0,
                tx_rate=0.0,
                timestamp=now,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "demo_rl.pt"
            rl_training.fit(trades, snaps, model_path=model_path, algo="ppo")
            model = LightningPPO()
            model.load_state_dict(torch.load(model_path))
            dataset = _TradeDataset(trades, snaps, sims_per_token=0)
            loader = DataLoader(dataset, batch_size=len(dataset))
            model.eval()
            total = 0.0
            with torch.no_grad():
                for states, actions, rewards in loader:
                    logits = model.actor(states)
                    preds = logits.argmax(dim=1)
                    mask = preds == actions
                    if mask.any():
                        total += float(rewards[mask].sum())
            return float(total)
    except Exception:  # pragma: no cover - best effort
        return 0.0
    finally:
        simulation.predict_price_movement = orig_predict


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
        "--preset",
        choices=sorted(PRESET_DATA_FILES.keys()),
        default="short",
        help="Bundled price dataset to use (default: 'short')",
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
    parser.add_argument(
        "--full-system",
        action="store_true",
        help="Run the heavier RL components",
    )
    args = parser.parse_args(argv)

    preset = args.preset
    if args.data is not None:
        if args.preset != parser.get_default("preset"):
            raise ValueError("Cannot specify both --data and --preset")
        preset = None

    global FULL_SYSTEM
    FULL_SYSTEM = bool(args.full_system)

    loaded = load_prices(args.data, preset)
    if isinstance(loaded, dict):
        price_map = loaded
    else:
        # Default token name for single-token datasets
        price_map = {"demo": loaded}
    multi_token = len(price_map) > 1

    first_token, (prices, dates) = next(iter(price_map.items()))
    if not prices or not dates:
        raise ValueError("price data must contain at least one entry")

    # Demonstrate Memory usage and portfolio hedging using the first token.
    # ``Memory`` relies on SQLAlchemy which may not be installed in minimal
    # environments.  Attempt to instantiate it and fall back to a simple
    # in-memory implementation when unavailable.
    try:
        if Memory is None:
            raise ImportError
        mem = Memory("sqlite:///:memory:")  # type: ignore[call-arg]
    except ImportError:
        from .simple_memory import SimpleMemory

        mem = SimpleMemory()

    async def _record_demo_trade() -> None:
        await mem.log_trade(
            token=first_token, direction="buy", amount=1.0, price=prices[0]
        )
        trades = await mem.list_trades(token=first_token)
        assert len(trades) == 1

    asyncio.run(_record_demo_trade())

    mem.log_var(0.0)
    asyncio.run(mem.close())

    # Compute correlations between strategy returns for the first token
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
    hedged_weights = hedge_allocation(
        {"buy_hold": 1.0, "momentum": 0.0}, corr_pairs
    )

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

    # Persist correlation pairs and hedged weights for inspection
    corr_out = {str((a, b)): c for (a, b), c in corr_pairs.items()}
    with open(args.reports / "correlations.json", "w", encoding="utf-8") as cf:
        json.dump(corr_out, cf, indent=2)
    with open(args.reports / "hedged_weights.json", "w", encoding="utf-8") as hf:
        json.dump(hedged_weights, hf, indent=2)

    # Produce concise summaries for stdout and later highlights.json
    top_corr = sorted(corr_pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    corr_summary = {f"{a}-{b}": c for (a, b), c in top_corr}
    if corr_summary:
        print(
            "Key correlations:",
            ", ".join(f"{pair}: {val:.2f}" for pair, val in corr_summary.items()),
        )
    if hedged_weights:
        print(
            "Hedged weights:",
            ", ".join(f"{k}: {v:.2f}" for k, v in hedged_weights.items()),
        )
    summary: List[Dict[str, float | int | str]] = []
    trade_history: List[Dict[str, float | str]] = []

    for token, (prices, dates) in price_map.items():
        if not prices or not dates:
            raise ValueError("price data must contain at least one entry")
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
                "token": token,
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

            # record per-period capital history for this strategy/token
            capital = args.capital
            trade_history.append(
                {
                    "token": token,
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
                        "token": token,
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
                        "token": token,
                        "strategy": name,
                        "period": i,
                        "date": dates[i],
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
                plt.savefig(args.reports / f"{token}_{name}.png")
                plt.close()
            except Exception as exc:  # pragma: no cover - plotting optional
                print(f"Plotting failed for {token} {name}: {exc}")

    json_path = args.reports / "summary.json"
    csv_path = args.reports / "summary.csv"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=summary[0].keys())
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    # Derive an aggregate view across tokens using the best strategy for each
    # token.  ``summary`` holds a row for every (token, strategy) pair so we
    # select the strategy with the highest final capital per token and compute
    # global metrics from those winners.
    per_token_best: Dict[str, Dict[str, float | int | str]] = {}
    for row in summary:
        tok = str(row["token"])
        existing = per_token_best.get(tok)
        if existing is None or float(row["final_capital"]) > float(existing["final_capital"]):
            per_token_best[tok] = row

    agg_rows = [
        {
            "token": t,
            "strategy": r["config"],
            "roi": r["roi"],
            "sharpe": r["sharpe"],
            "final_capital": r["final_capital"],
        }
        for t, r in per_token_best.items()
    ]

    global_roi = (
        sum(float(r["roi"]) for r in agg_rows) / len(agg_rows) if agg_rows else 0.0
    )
    global_sharpe = (
        sum(float(r["sharpe"]) for r in agg_rows) / len(agg_rows)
        if agg_rows
        else 0.0
    )

    top = max(summary, key=lambda e: e["final_capital"]) if summary else None

    aggregate: Dict[str, object] = {
        "global_roi": global_roi,
        "global_sharpe": global_sharpe,
        "per_token": agg_rows,
    }
    if top is not None:
        aggregate.update(
            {
                "top_token": top["token"],
                "top_strategy": top["config"],
                "top_final_capital": top["final_capital"],
                "top_roi": top["roi"],
                "top_sharpe": top["sharpe"],
            }
        )

    agg_json = args.reports / "aggregate_summary.json"
    agg_csv = args.reports / "aggregate_summary.csv"
    with open(agg_json, "w", encoding="utf-8") as af:
        json.dump(aggregate, af, indent=2)
    with open(agg_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["token", "strategy", "roi", "sharpe", "final_capital"]
        )
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    # Write trade history in CSV and JSON for inspection
    hist_csv = args.reports / "trade_history.csv"
    hist_json = args.reports / "trade_history.json"
    if trade_history:
        with open(hist_json, "w", encoding="utf-8") as jf:
            json.dump(trade_history, jf, indent=2)
        with open(hist_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(
                cf,
                fieldnames=[
                    "token",
                    "strategy",
                    "period",
                    "date",
                    "action",
                    "price",
                    "capital",
                ],
            )
            writer.writeheader()
            for row in trade_history:
                writer.writerow(row)

    # Exercise trade types via lightweight stubs
    async def _exercise_trade_types() -> Dict[str, object]:
        arb, fl_sig, sniped, pools = await asyncio.gather(
            _demo_arbitrage(),
            _demo_flash_loan(),
            _demo_sniper(),
            _demo_dex_scanner(),
        )
        rl_reward = _demo_rl_agent()
        arb_path = arb.get("path") if isinstance(arb, dict) else None
        arb_profit = arb.get("profit") if isinstance(arb, dict) else None
        return {
            "arbitrage_path": arb_path,
            "arbitrage_profit": arb_profit,
            "flash_loan_signature": fl_sig,
            "sniper_tokens": sniped,
            "dex_new_pools": pools,
            "rl_reward": rl_reward,
        }

    trade_outputs = asyncio.run(_exercise_trade_types())

    required = {"arbitrage", "flash_loan", "sniper", "dex_scanner"}
    missing = required - used_trade_types
    if missing:
        raise RuntimeError(
            f"Demo did not exercise trade types: {', '.join(sorted(missing))}"
        )

    # Collect resource usage metrics if available
    cpu_usage = None
    mem_pct = None
    if psutil is not None:
        try:
            cpu_usage = resource_monitor.get_cpu_usage()
            mem_pct = psutil.virtual_memory().percent
        except Exception:
            pass

    # Write highlights summarising top performing strategy and trade results
    if top is not None:
        highlights = {
            "top_strategy": top["config"],
            "top_final_capital": top["final_capital"],
            "top_roi": top["roi"],
        }
        if multi_token:
            highlights["top_token"] = top["token"]
        if cpu_usage is not None:
            highlights["cpu_usage"] = cpu_usage
        if mem_pct is not None:
            highlights["memory_percent"] = mem_pct
        highlights.update(trade_outputs)
        if corr_summary:
            highlights["key_correlations"] = corr_summary
        if hedged_weights:
            highlights["hedged_weights"] = hedged_weights
        with open(args.reports / "highlights.json", "w", encoding="utf-8") as hf:
            json.dump(highlights, hf, indent=2)

    # Display a simple capital summary on stdout
    print("Capital Summary:")
    for row in summary:
        # Include ROI and Sharpe ratio alongside final capital so that the CLI
        # output mirrors the contents of ``summary.json``.
        prefix = f"{row['token']} {row['config']}" if multi_token else row["config"]
        line = (
            f"{prefix}: {row['final_capital']:.2f} "
            f"ROI {row['roi']:.4f} "
            f"Sharpe {row['sharpe']:.4f} "
            f"Drawdown {row['drawdown']:.4f} "
            f"Win rate {row['win_rate']:.4f}"
        )
        print(line)
    if top:
        top_prefix = (
            f"{top['token']} {top['config']}" if multi_token else top["config"]
        )
        print(
            f"Top strategy: {top_prefix} with final capital {top['final_capital']:.2f}"
        )
    print("Trade type results:", json.dumps(trade_outputs))
    if cpu_usage is not None and mem_pct is not None:
        print(
            f"Resource usage - CPU: {cpu_usage:.2f}% Memory: {mem_pct:.2f}%"
        )

    print(f"Wrote reports to {args.reports}")


if __name__ == "__main__":  # pragma: no cover
    main()
