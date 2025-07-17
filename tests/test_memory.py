import pytest
from solhunter_zero.memory import Memory


def test_log_and_list_trades():
    mem = Memory('sqlite:///:memory:')
    mem.log_trade(token='TEST', direction='buy', amount=1.5, price=2.0)
    mem.log_trade(token='TEST2', direction='sell', amount=0.5, price=1.0)

    trades = mem.list_trades()
    assert len(trades) == 2
    assert trades[0].token == 'TEST'
    assert trades[0].direction == 'buy'
    assert trades[1].token == 'TEST2'
    assert trades[1].direction == 'sell'
