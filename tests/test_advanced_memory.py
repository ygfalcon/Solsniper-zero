from solhunter_zero.advanced_memory import AdvancedMemory


def test_insert_search_persist(tmp_path):
    db = tmp_path / "mem.db"
    idx = tmp_path / "index.faiss"
    mem = AdvancedMemory(url=f"sqlite:///{db}", index_path=str(idx))
    sim_id = mem.log_simulation("TOK", expected_roi=1.5, success_prob=0.8)
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
