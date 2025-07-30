import json
import os
import asyncio
import aiohttp
import time
import pytest

from solhunter_zero import depth_client

import types
import sys

dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
sys.modules.setdefault("transformers", dummy_trans)
dummy_solana = types.ModuleType("solana")
dummy_rpc = types.ModuleType("solana.rpc")
dummy_api = types.ModuleType("solana.rpc.api")
dummy_api.Client = object
dummy_async_api = types.ModuleType("solana.rpc.async_api")
dummy_async_api.AsyncClient = object
sys.modules.setdefault("solana", dummy_solana)
sys.modules.setdefault("solana.rpc", dummy_rpc)
sys.modules.setdefault("solana.rpc.api", dummy_api)
sys.modules.setdefault("solana.rpc.async_api", dummy_async_api)
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
dummy_sk_linear = types.ModuleType("sklearn.linear_model")
dummy_sk_linear.LinearRegression = object
sys.modules.setdefault("sklearn.linear_model", dummy_sk_linear)
dummy_sk_ensemble = types.ModuleType("sklearn.ensemble")
dummy_sk_ensemble.GradientBoostingRegressor = object
dummy_sk_ensemble.RandomForestRegressor = object
sys.modules.setdefault("sklearn.ensemble", dummy_sk_ensemble)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("xgboost", types.ModuleType("xgboost"))
sys.modules.setdefault("solders", types.ModuleType("solders"))
sys.modules.setdefault("solders.keypair", types.ModuleType("solders.keypair"))
sys.modules["solders.keypair"].Keypair = object
sys.modules.setdefault("solders.transaction", types.ModuleType("solders.transaction"))
sys.modules["solders.transaction"].VersionedTransaction = object
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch"].Tensor = object
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules["torch.nn"].Module = object

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

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: FakeSession())
    monkeypatch.setattr(data_sync, "fetch_sentiment", lambda *a, **k: 0.0)

    await data_sync.sync_snapshots(["TOK"], db_path=str(db), limit_gb=0.0000001, base_url="http://api")

    data = OfflineData(f"sqlite:///{db}")
    snaps = await data.list_snapshots("TOK")
    assert not snaps  # pruned due to low limit
    assert called["urls"]


@pytest.mark.asyncio
async def test_depth_snapshot_listener(tmp_path, monkeypatch):
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

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: FakeSession([msg]))

    db = tmp_path / "data.db"
    data = OfflineData(f"sqlite:///{db}")
    from solhunter_zero.data_pipeline import start_depth_snapshot_listener

    unsub = start_depth_snapshot_listener(data)
    await depth_client.listen_depth_ws(max_updates=1)
    unsub()

    snaps = await data.list_snapshots("tok")
    assert snaps and snaps[0].price == 1.0


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

    monkeypatch.setattr("aiohttp.ClientSession", lambda *a, **k: FakeSession())
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

