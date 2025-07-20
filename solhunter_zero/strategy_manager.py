from __future__ import annotations

from typing import Iterable, List, Tuple

from .backtester import StrategyFunc, backtest_strategies, DEFAULT_STRATEGIES


class StrategyManager:
    """Select the best strategy based on recent price history."""

    def __init__(
        self, strategies: Iterable[Tuple[str, StrategyFunc]] | None = None
    ) -> None:
        self.strategies = list(strategies) if strategies is not None else list(DEFAULT_STRATEGIES)
        self.current_name = self.strategies[0][0] if self.strategies else None

    def select(self, prices: List[float]) -> None:
        """Run backtests and select the best performing strategy."""

        results = backtest_strategies(prices, self.strategies)
        if results:
            self.current_name = results[0].name

    @property
    def current_strategy(self) -> StrategyFunc | None:
        for name, fn in self.strategies:
            if name == self.current_name:
                return fn
        return None
