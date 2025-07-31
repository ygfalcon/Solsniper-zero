from solhunter_zero.advanced_memory import AdvancedMemory


def test_insert_search_persist(tmp_path):
    db = tmp_path / "mem.db"
    idx = tmp_path / "index.faiss"
    mem = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx))
    sim_id = mem.log_simulation("TOK", expected_roi=1.5, success_prob=0.8, agent="test")
    mem.log_trade(
        token="TOK",
        direction="buy",
        amount=1.0,
        price=2.0,
        reason="test",
        context="great momentum ahead",
        emotion="bullish",
        simulation_id=sim_id,
    )

    trades = mem.list_trades()
    assert len(trades) == 1
    assert trades[0].context == "great momentum ahead"

    results = mem.search("momentum")
    assert results and results[0].id == trades[0].id

    # Recreate memory to test persistence
    mem2 = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx))
    trades2 = mem2.list_trades()
    assert len(trades2) == 1
    results2 = mem2.search("momentum")
    assert results2 and results2[0].id == trades[0].id


def test_advanced_list_filters(tmp_path):
    db = tmp_path / "mem.db"
    idx = tmp_path / "idx.faiss"
    mem = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx))
    a = mem.log_trade(token="X", direction="buy", amount=1, price=1)
    b = mem.log_trade(token="Y", direction="sell", amount=2, price=1)
    c = mem.log_trade(token="X", direction="sell", amount=3, price=1)

    assert len(mem.list_trades(token="X")) == 2
    assert len(mem.list_trades(limit=2)) == 2
    ids = [t.id for t in mem.list_trades(since_id=b)]
    assert ids == [c]


def test_replicated_trade_dedup(tmp_path):
    db = tmp_path / "rep.db"
    idx = tmp_path / "rep.index"
    mem = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx), replicate=True)

    from uuid import uuid4
    from solhunter_zero.event_bus import publish
    from solhunter_zero.schemas import TradeLogged

    tid = str(uuid4())
    event = TradeLogged(token="TOK", direction="buy", amount=1.0, price=1.0, uuid=tid)
    publish("trade_logged", event)
    publish("trade_logged", event)

    trades = mem.list_trades()
    assert len(trades) == 1
    assert trades[0].uuid == tid


def test_memory_sync_exchange(tmp_path):
    db1 = tmp_path / "a.db"
    idx1 = tmp_path / "a.index"
    db2 = tmp_path / "b.db"
    idx2 = tmp_path / "b.index"

    mem1 = AdvancedMemory(url=f"sqlite:///{db1}", index_path=str(idx1), replicate=True)
    mem2 = AdvancedMemory(url=f"sqlite:///{db2}", index_path=str(idx2), replicate=True)

    mem1.log_trade(token="A", direction="buy", amount=1.0, price=1.0)

    from solhunter_zero.event_bus import publish
    from solhunter_zero.schemas import MemorySyncRequest

    publish("memory_sync_request", MemorySyncRequest(last_id=0))

    assert len(mem2.list_trades()) == 1

    mem2.log_trade(token="B", direction="sell", amount=2.0, price=2.0)
    publish("memory_sync_request", MemorySyncRequest(last_id=0))

    assert len(mem1.list_trades()) == 2
    assert len(mem2.list_trades()) == 2
