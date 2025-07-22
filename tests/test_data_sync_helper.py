import json
from pathlib import Path
import os
import types

import solhunter_zero.data_sync as data_sync
from solhunter_zero.offline_data import OfflineData


def test_sync_snapshots_and_prune(tmp_path, monkeypatch):
    db = tmp_path / "data.db"

    called = {}

    def fake_get(url, timeout=10):
        called.setdefault("urls", []).append(url)
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"snapshots": [
                    {"price": 1.0, "depth": 2.0, "imbalance": 0.1},
                    {"price": 1.1, "depth": 2.1, "imbalance": 0.2},
                ]}
        return Resp()

    monkeypatch.setattr(data_sync, "requests", types.SimpleNamespace(get=fake_get))

    data_sync.sync_snapshots(["TOK"], db_path=str(db), limit_gb=0.0000001, base_url="http://api")

    data = OfflineData(f"sqlite:///{db}")
    snaps = data.list_snapshots("TOK")
    assert not snaps  # pruned due to low limit
    assert called["urls"]

