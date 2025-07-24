import json
from pathlib import Path
import os

import solhunter_zero.data_sync as data_sync
from solhunter_zero.offline_data import OfflineData


def test_sync_snapshots_and_prune(tmp_path, monkeypatch):
    db = tmp_path / "data.db"

    called = {}

    class FakeResp:
        def __init__(self, url):
            called.setdefault("urls", []).append(url)
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def raise_for_status(self):
            pass
        async def json(self):
            return {"snapshots": [
                {"price": 1.0, "depth": 2.0, "total_depth": 3.0, "imbalance": 0.1},
                {"price": 1.1, "depth": 2.1, "total_depth": 3.1, "imbalance": 0.2},
            ]}

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def get(self, url, timeout=10):
            return FakeResp(url)

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())

    data_sync.sync_snapshots(["TOK"], db_path=str(db), limit_gb=0.0000001, base_url="http://api")

    data = OfflineData(f"sqlite:///{db}")
    snaps = data.list_snapshots("TOK")
    assert not snaps  # pruned due to low limit
    assert called["urls"]

