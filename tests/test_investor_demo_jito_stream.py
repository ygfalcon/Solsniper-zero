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


def test_jito_stream_in_highlights(tmp_path):
    investor_demo.main(["--reports", str(tmp_path), "--preset", "short"])
    highlights = json.loads((tmp_path / "highlights.json").read_text())
    swaps = highlights.get("jito_swaps")
    assert swaps and swaps[0]["token"] == "tok"
    assert swaps[0]["size"] == 1.0
