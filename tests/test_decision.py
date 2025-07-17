from solhunter_zero.decision import should_buy
from solhunter_zero.simulation import SimulationResult


def test_should_buy_empty():
    assert should_buy([]) is False


def test_should_buy_positive():
    sims = [
        SimulationResult(success_prob=0.7, expected_roi=1.2),
        SimulationResult(success_prob=0.65, expected_roi=1.5),
    ]
    assert should_buy(sims) is True


def test_should_buy_negative():
    sims = [
        SimulationResult(success_prob=0.5, expected_roi=0.8),
        SimulationResult(success_prob=0.6, expected_roi=0.9),
    ]
    assert should_buy(sims) is False
