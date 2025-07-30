import pytest
from solhunter_zero.memory import Memory


import pytest


@pytest.mark.asyncio
async def test_log_and_list_trades():
    mem = Memory('sqlite:///:memory:')
    await mem.log_trade(token='TEST', direction='buy', amount=1.5, price=2.0)
    await mem.log_trade(token='TEST2', direction='sell', amount=0.5, price=1.0)

    trades = await mem.list_trades()
    assert len(trades) == 2
    assert trades[0].token == 'TEST'
    assert trades[0].direction == 'buy'
    assert trades[1].token == 'TEST2'
    assert trades[1].direction == 'sell'


@pytest.mark.asyncio
async def test_list_trades_filters():
    mem = Memory('sqlite:///:memory:')
    a = await mem.log_trade(token='A', direction='buy', amount=1.0, price=1.0)
    b = await mem.log_trade(token='B', direction='sell', amount=1.0, price=1.0)
    c = await mem.log_trade(token='A', direction='sell', amount=2.0, price=1.0)

    assert len(await mem.list_trades(token='A')) == 2
    assert len(await mem.list_trades(limit=2)) == 2
    ids = [t.id for t in await mem.list_trades(since_id=b)]
    assert ids == [c]


@pytest.mark.asyncio
async def test_log_and_list_vars():
    mem = Memory('sqlite:///:memory:')
    await mem.log_var(0.1)
    await mem.log_var(0.2)
    vals = await mem.list_vars()
    assert [v.value for v in vals] == [0.1, 0.2]


def test_trade_replication_event(tmp_path):
    db = tmp_path / "rep.db"
    idx = tmp_path / "rep.index"
    from solhunter_zero.advanced_memory import AdvancedMemory
    from solhunter_zero.event_bus import publish
    from solhunter_zero.schemas import TradeLogged

    mem = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx), replicate=True)
    publish(
        "trade_logged",
        TradeLogged(token="TOK", direction="buy", amount=1.0, price=1.0),
    )
    trades = mem.list_trades()
    assert trades and trades[0].token == "TOK"
