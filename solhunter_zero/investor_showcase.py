from __future__ import annotations

"""Standalone investor showcase.

This module runs all bundled strategies on a dummy dataset and prints the
resulting actions. It is intended for quick offline demonstrations and exits
with the status code of a small validation test suite.
"""

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import List, Dict, Any

import pytest

from .strategy_manager import StrategyManager
from .datasets.sample_ticks import load_sample_ticks


class DummyPortfolio:
    def __init__(self) -> None:
        self.balances: Dict[str, Any] = {}

    def position_roi(self, token: str, price: float) -> float:
        return 0.0

    def update_drawdown(self, prices: Dict[str, float]) -> None:
        return None

    def current_drawdown(self, prices: Dict[str, float]) -> float:
        return 0.0

    def total_value(self, prices: Dict[str, float]) -> float:
        return 100.0

    def percent_allocated(self, token: str, prices: Dict[str, float]) -> float:
        return 0.0


def _install_stub_strategies(price: float) -> None:
    async def sniper_eval(token: str, _portfolio) -> List[Dict[str, Any]]:  # pragma: no cover - demo
        return [{"token": token, "side": "buy", "amount": 1.0, "price": price}]

    async def arbitrage_eval(token: str, _portfolio) -> List[Dict[str, Any]]:  # pragma: no cover - demo
        action = {"token": token, "amount": 1.0, "price": 0.0}
        return [dict(action, side="buy"), dict(action, side="sell")]

    sniper_mod = types.ModuleType("solhunter_zero.sniper")
    sniper_mod.evaluate = sniper_eval  # type: ignore[attr-defined]
    arbitrage_mod = types.ModuleType("solhunter_zero.arbitrage")
    arbitrage_mod.evaluate = arbitrage_eval  # type: ignore[attr-defined]

    sys.modules["solhunter_zero.sniper"] = sniper_mod
    sys.modules["solhunter_zero.arbitrage"] = arbitrage_mod


def run_showcase() -> List[Dict[str, Any]]:
    """Execute all strategies on dummy data and return their actions."""

    ticks = load_sample_ticks()
    price = float(ticks[0]["price"]) if ticks else 1.0

    _install_stub_strategies(price)

    portfolio = DummyPortfolio()
    manager = StrategyManager()

    token = "DUMMY"
    actions = asyncio.run(manager.evaluate(token, portfolio))
    print(json.dumps(actions, indent=2))
    Path("showcase_actions.json").write_text(json.dumps(actions, indent=2))
    return actions


def main() -> int:
    run_showcase()
    return pytest.main(["tests/staging/test_investor_showcase.py"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
