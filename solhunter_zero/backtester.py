from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np


StrategyFunc = Callable[[List[float]], List[float]]


@dataclass
class StrategyResult:
    """Result of a backtest for a single strategy."""

    name: str
    roi: float
    sharpe: float


def buy_and_hold(prices: List[float]) -> List[float]:
    """Return daily returns for a buy and hold approach."""

    return [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] > 0
    ]


def momentum(prices: List[float]) -> List[float]:
    """Return returns when buying only on positive momentum."""

    returns: List[float] = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i - 1]) / prices[i - 1]
        if r > 0:
            returns.append(r)
    return returns


DEFAULT_STRATEGIES: List[Tuple[str, StrategyFunc]] = [
    ("buy_hold", buy_and_hold),
    ("momentum", momentum),
]


def backtest_strategies(
    prices: List[float],
    strategies: Iterable[Tuple[str, StrategyFunc]] | None = None,
) -> List[StrategyResult]:
    """Backtest ``strategies`` on ``prices`` and return ranked results."""

    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    results: List[StrategyResult] = []
    for name, strat in strategies:
        rets = strat(prices)
        if not rets:
            roi = 0.0
            sharpe = 0.0
        else:
            arr = np.array(rets, dtype=float)
            roi = float(np.prod(1 + arr) - 1)
            std = float(arr.std())
            mean = float(arr.mean())
            sharpe = mean / std if std > 0 else 0.0
        results.append(StrategyResult(name=name, roi=roi, sharpe=sharpe))

    results.sort(key=lambda r: (r.roi, r.sharpe), reverse=True)
    return results
