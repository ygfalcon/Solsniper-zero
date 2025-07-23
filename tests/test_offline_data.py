from solhunter_zero.offline_data import OfflineData


def test_offline_data_roundtrip(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    data.log_snapshot("tok", 1.0, 2.0, 0.5, 0.0, 0.0)
    snaps = data.list_snapshots("tok")
    assert snaps and snaps[0].token == "tok"
    assert snaps[0].tx_rate == 0.0
    assert snaps[0].whale_share == 0.0
    assert snaps[0].spread == 0.0
    data.log_trade("tok", "buy", 1.0, 2.0)
    trades = data.list_trades("tok")
    assert trades and trades[0].side == "buy"
