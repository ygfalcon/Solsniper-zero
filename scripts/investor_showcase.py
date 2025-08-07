#!/usr/bin/env python
"""Run bundled strategies on dummy price data.

This script demonstrates the
:class:`solhunter_zero.strategy_manager.StrategyManager` by loading a small
multi-token price dataset and executing several trading strategies. Results
from all strategies are merged and the cumulative return for each demo
strategy is printed.

The default strategies (:mod:`solhunter_zero.sniper` and
``solhunter_zero.arbitrage``) are included along with lightweight wrappers for
the investor demo strategies.  The wrappers expose an ``evaluate`` function so
they can be managed by :class:`StrategyManager`.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

from solhunter_zero.strategy_manager import StrategyManager
from solhunter_zero.investor_demo import (
    DEFAULT_STRATEGIES as DEMO_STRATS,
    load_prices,
)


# Global mapping used by demo strategy modules.
PRICE_DATA: Dict[str, Tuple[List[float], List[str]]] = {}


def _create_demo_module(name: str, func) -> str:
    """Expose ``func`` as a StrategyManager-compatible module.

    Parameters
    ----------
    name:
        Name of the strategy as defined in
        ``investor_demo.DEFAULT_STRATEGIES``.
    func:
        Function returning a list of returns for a sequence of prices.

    Returns
    -------
    str
        Fully-qualified module name registered with :mod:`sys.modules`.
    """

    module_name = "solhunter_zero._demo_" + name
    mod = types.ModuleType(module_name)

    def evaluate(token: str, _portfolio) -> List[Dict[str, float]]:
        prices, _dates = PRICE_DATA.get(token, ([], []))
        if not prices:
            return []
        returns = func(prices)
        total = sum(float(r) for r in returns)
        side = "buy" if total >= 0 else "sell"
        return [
            {
                "token": token,
                "side": side,
                "amount": abs(total),
                "price": prices[-1],
            }
        ]

    mod.evaluate = evaluate  # type: ignore[attr-defined]
    sys.modules[module_name] = mod
    return module_name


async def run_demo() -> None:
    """Load price data and execute all strategies."""

    data_path = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "data"
        / "prices_multitoken.json"
    )
    prices = load_prices(path=data_path)

    global PRICE_DATA
    PRICE_DATA = prices  # store for demo modules

    extra_modules = [
        _create_demo_module(name, func) for name, func in DEMO_STRATS
    ]

    manager = StrategyManager(
        StrategyManager.DEFAULT_STRATEGIES + extra_modules
    )

    from solhunter_zero.portfolio import Portfolio

    portfolio = Portfolio(path=None)
    merged_actions: List[Dict[str, float]] = []
    returns: Dict[str, float] = {name: 0.0 for name, _ in DEMO_STRATS}

    for token in prices.keys():
        actions = await manager.evaluate(token, portfolio)
        merged_actions.extend(actions)

        for name, func in DEMO_STRATS:
            rets = func(prices[token][0])
            returns[name] += sum(float(r) for r in rets)

    print("Merged actions:")
    for action in merged_actions:
        print(f"  {action}")

    print("\nPer-strategy returns:")
    for name, value in returns.items():
        print(f"  {name}: {value:.4f}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
