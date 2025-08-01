import json
import os
import asyncio
import aiohttp
import time
import pytest

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "pipeline"):
    transformers.pipeline = lambda *a, **k: lambda x: []
pytest.importorskip("sklearn")

from solhunter_zero import depth_client

import solhunter_zero.data_sync as data_sync
from solhunter_zero.offline_data import OfflineData


@pytest.mark.asyncio
async def test_sync_snapshots_and_prune(tmp_path, monkeypatch):
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
    monkeypatch.setattr(data_sync, "fetch_sentiment", lambda *a, **k: 0.0)

    await data_sync.sync_snapshots(["TOK"], db_path=str(db), limit_gb=0.0000001, base_url="http://api")

    data = OfflineData(f"sqlite:///{db}")
    snaps = await data.list_snapshots("TOK")
    assert not snaps  # pruned due to low limit
    assert called["urls"]


@pytest.mark.asyncio
async def test_depth_snapshot_listener(tmp_path, monkeypatch):
    msg = {
        "tok": {
            "price": 1.0,
            "depth": 2.0,
            "total_depth": 2.5,
            "imbalance": 0.1,
            "slippage": 0.2,
            "volume": 3.0,
            "tx_rate": 4.0,
            "whale_share": 0.3,
            "spread": 0.4,
            "sentiment": 0.5,
        }
    }

    async def fake_listener(*_a, **_k):
        from solhunter_zero.event_bus import publish

        publish("depth_service_status", {"status": "connected"}, _broadcast=False)
        publish("depth_update", msg, _broadcast=False)
        publish("depth_service_status", {"status": "disconnected"}, _broadcast=False)

    monkeypatch.setattr(depth_client, "listen_depth_ws", fake_listener)

    db = tmp_path / "data.db"
    data = OfflineData(f"sqlite:///{db}")
    from solhunter_zero.data_pipeline import start_depth_snapshot_listener

    unsub = start_depth_snapshot_listener(data)
    await depth_client.listen_depth_ws(max_updates=1)
    unsub()

    snaps = await data.list_snapshots("tok")
    assert snaps
    snap = snaps[0]
    for field, value in msg["tok"].items():
        assert pytest.approx(getattr(snap, field)) == value


def test_sync_concurrency(tmp_path, monkeypatch):
    db = tmp_path / "data.db"

    class FakeResp:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def raise_for_status(self):
            pass
        async def json(self):
            await asyncio.sleep(0.05)
            return {"snapshots": []}

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def get(self, url, timeout=10):
            return FakeResp()

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    monkeypatch.setattr(data_sync, "fetch_sentiment", lambda *a, **k: 0.0)

    async def run(conc):
        start = time.perf_counter()
        await data_sync.sync_snapshots([
            "A",
            "B",
        ], db_path=str(db), base_url="http://api", concurrency=conc)
        return time.perf_counter() - start

    t1 = asyncio.run(run(1))
    t2 = asyncio.run(run(2))
    assert t2 < t1

