from solhunter_zero.offline_data import OfflineData


def test_offline_data_roundtrip(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    data.log_snapshot("tok", 1.0, 2.0, 0.5)
    snaps = data.list_snapshots("tok")
    assert snaps and snaps[0].token == "tok"
