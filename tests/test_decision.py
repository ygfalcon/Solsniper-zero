from solhunter_zero.decision import should_buy, should_sell
from solhunter_zero.simulation import SimulationResult


def test_should_buy_empty():
    assert should_buy([]) is False


def test_should_buy_positive():
    sims = [
        SimulationResult(success_prob=0.7, expected_roi=1.2, volume=200.0, liquidity=500.0),
        SimulationResult(success_prob=0.65, expected_roi=1.5, volume=200.0, liquidity=500.0),
    ]
    assert should_buy(sims) is True


def test_should_buy_negative():
    sims = [
        SimulationResult(success_prob=0.5, expected_roi=0.8, volume=200.0, liquidity=500.0),
        SimulationResult(success_prob=0.6, expected_roi=0.9, volume=200.0, liquidity=500.0),
    ]
    assert should_buy(sims) is False


def test_should_buy_high_thresholds():
    sims = [
        SimulationResult(success_prob=0.7, expected_roi=1.2, volume=200.0, liquidity=500.0),
        SimulationResult(success_prob=0.65, expected_roi=1.5, volume=200.0, liquidity=500.0),
    ]
    # require very high sharpe ratio
    assert should_buy(sims, min_sharpe=10.0) is False


def test_should_buy_volume_threshold():
    sims = [SimulationResult(success_prob=0.8, expected_roi=1.1, volume=50.0, liquidity=100.0)]
    assert should_buy(sims, min_volume=100.0) is False
    assert should_buy(sims, min_volume=40.0, min_sharpe=0.0) is True


def test_should_sell_negative_roi():
    sims = [
        SimulationResult(success_prob=0.5, expected_roi=-0.1, volume=200.0, liquidity=500.0),
        SimulationResult(success_prob=0.45, expected_roi=-0.05, volume=200.0, liquidity=500.0),
    ]
    assert should_sell(sims) is True


def test_should_sell_low_success():
    sims = [
        SimulationResult(success_prob=0.3, expected_roi=0.2, volume=200.0, liquidity=500.0),
        SimulationResult(success_prob=0.35, expected_roi=0.1, volume=200.0, liquidity=500.0),
    ]
    assert should_sell(sims) is True


def test_should_sell_positive_outlook():
    sims = [SimulationResult(success_prob=0.8, expected_roi=0.5, volume=200.0, liquidity=500.0)]
    assert should_sell(sims) is False
