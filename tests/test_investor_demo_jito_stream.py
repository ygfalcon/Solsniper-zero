import asyncio
import json

import pytest

from solhunter_zero import investor_demo
from solhunter_zero.event_bus import subscribe


@pytest.fixture(autouse=True)
def clear_used_trade_types():
    investor_demo.used_trade_types.clear()
    yield
    investor_demo.used_trade_types.clear()


def test_demo_jito_stream_events():
    events = []
    unsub = subscribe("pending_swap", lambda p: events.append(p))
    swaps = asyncio.run(investor_demo._demo_jito_stream())
    unsub()

    assert swaps == events
    assert swaps == [
        {"token": "tok", "address": "tok", "size": 1.0, "slippage": 0.1}
    ]
    assert investor_demo.used_trade_types == {"jito_stream"}


def test_jito_stream_in_highlights(tmp_path, monkeypatch):
    async def fake_arb():
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": [], "profit": 0.0}

    async def fake_flash():
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper():
        investor_demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def fake_dex():
        investor_demo.used_trade_types.add("dex_scanner")
        return ["pool"]

    async def fake_route():
        investor_demo.used_trade_types.add("route_ffi")
        return {"path": [], "profit": 0.0}

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arb)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route)

    investor_demo.main(["--reports", str(tmp_path), "--preset", "full"])
    highlights = json.loads((tmp_path / "highlights.json").read_text())
    swaps = highlights.get("jito_swaps")
    assert swaps and swaps[0]["token"] == "tok"
    assert swaps[0]["size"] == 1.0
