import json
from pathlib import Path
import os
import asyncio
import aiohttp

from solhunter_zero import depth_client

import types
import sys

dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
sys.modules.setdefault("transformers", dummy_trans)

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
    monkeypatch.setattr(data_sync, "fetch_sentiment", lambda *a, **k: 0.0)

    data_sync.sync_snapshots(["TOK"], db_path=str(db), limit_gb=0.0000001, base_url="http://api")

    data = OfflineData(f"sqlite:///{db}")
    snaps = data.list_snapshots("TOK")
    assert not snaps  # pruned due to low limit
    assert called["urls"]


def test_depth_snapshot_listener(tmp_path, monkeypatch):
    msg = {"tok": {"price": 1.0, "depth": 2.0, "total_depth": 2.0, "imbalance": 0.1}}

    class FakeMsg:
        def __init__(self, data):
            self.type = aiohttp.WSMsgType.TEXT
            self.data = json.dumps(data)

    class FakeWS:
        def __init__(self, messages):
            self.messages = list(messages)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.messages:
                return FakeMsg(self.messages.pop(0))
            raise StopAsyncIteration

    class FakeSession:
        def __init__(self, messages):
            self.messages = messages

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def ws_connect(self, url):
            self.url = url
            return FakeWS(self.messages)

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession([msg]))

    db = tmp_path / "data.db"
    data = OfflineData(f"sqlite:///{db}")
    from solhunter_zero.data_pipeline import start_depth_snapshot_listener

    unsub = start_depth_snapshot_listener(data)
    asyncio.run(depth_client.listen_depth_ws(max_updates=1))
    unsub()

    snaps = data.list_snapshots("tok")
    assert snaps and snaps[0].price == 1.0

