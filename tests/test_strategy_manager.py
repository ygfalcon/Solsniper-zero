import numpy as np
from solhunter_zero import backtester
from solhunter_zero.strategy_manager import StrategyManager


def test_backtester_ranking():
    def good(prices):
        return [0.1 for _ in prices[1:]]

    def bad(prices):
        return [-0.1 for _ in prices[1:]]

    prices = [1, 2, 3]
    results = backtester.backtest_strategies(
        prices, [("good", good), ("bad", bad)]
    )
    assert results[0].name == "good"


def test_strategy_manager_selects_best():
    def strat_a(prices):
        return [0.05 for _ in prices[1:]]

    def strat_b(prices):
        return [-0.02 for _ in prices[1:]]

    manager = StrategyManager([("a", strat_a), ("b", strat_b)])
    manager.select([1, 2, 3, 4])
    assert manager.current_strategy is strat_a

