import asyncio
import sys
import types
import pytest

from solhunter_zero import investor_demo


@pytest.fixture(autouse=True)
def clear_used_trade_types():
    investor_demo.used_trade_types.clear()
    yield
    investor_demo.used_trade_types.clear()


def test_route_and_jito(monkeypatch):
    routeffi = types.SimpleNamespace(
        _best_route_json=lambda *a, **k: (["x", "y"], 1.0)
    )

    jito_mod = types.ModuleType("jito_stream")

    async def stream_pending_transactions(_url, *, auth=None):
        yield {
            "pendingTransactions": [
                {"swap": {"token": "tok", "size": 1.0, "slippage": 0.1}}
            ]
        }

    async def stream_pending_swaps(url, *, auth=None):
        async for data in jito_mod.stream_pending_transactions(url, auth=auth):
            for tx in data["pendingTransactions"]:
                swap = tx["swap"]
                yield {
                    "token": swap["token"],
                    "address": swap["token"],
                    "size": swap["size"],
                    "slippage": swap["slippage"],
                }

    jito_mod.stream_pending_transactions = stream_pending_transactions
    jito_mod.stream_pending_swaps = stream_pending_swaps

    monkeypatch.setitem(sys.modules, "solhunter_zero.routeffi", routeffi)
    monkeypatch.setitem(sys.modules, "solhunter_zero.jito_stream", jito_mod)

    route = asyncio.run(investor_demo._demo_route_ffi())
    swaps = asyncio.run(investor_demo._demo_jito_stream())

    assert route["path"] == ["x", "y"]
    assert swaps == [
        {"token": "tok", "address": "tok", "size": 1.0, "slippage": 0.1}
    ]
    assert investor_demo.used_trade_types == {"route_ffi", "jito_stream"}
