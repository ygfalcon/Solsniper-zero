import asyncio
import json
import importlib.util
import sys

# Restore real websockets implementation if stubs are present
sys.modules.pop("websockets", None)
spec = importlib.util.find_spec("websockets")
assert spec and spec.loader
websockets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(websockets)
sys.modules["websockets"] = websockets

import pytest

from solhunter_zero import investor_demo
from solhunter_zero.event_bus import subscribe


@pytest.mark.asyncio
async def test_investor_demo_streaming(tmp_path):
    closed1 = asyncio.Event()
    closed2 = asyncio.Event()

    async def orca_handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"token": "TOK", "price": 1.0}))
        await ws.wait_closed()
        closed1.set()

    async def ray_handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"token": "TOK", "price": 2.0}))
        await ws.wait_closed()
        closed2.set()

    server1 = await websockets.serve(orca_handler, "localhost", 0)
    port1 = server1.sockets[0].getsockname()[1]
    server2 = await websockets.serve(ray_handler, "localhost", 0)
    port2 = server2.sockets[0].getsockname()[1]

    events = []
    unsub = subscribe("price_update", lambda p: events.append(p))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        investor_demo.main,
        [
            "--price-streams",
            f"orca=ws://localhost:{port1},raydium=ws://localhost:{port2}",
            "--tokens",
            "TOK",
            "--reports",
            str(tmp_path),
        ],
    )

    unsub()

    assert any(e.get("venue") == "orca" for e in events)
    assert any(e.get("venue") == "raydium" for e in events)
    assert closed1.is_set()
    assert closed2.is_set()

    server1.close()
    await server1.wait_closed()
    server2.close()
    await server2.wait_closed()
