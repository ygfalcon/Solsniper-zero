import asyncio
import json
import pytest
import sys
import types
import importlib.util
import importlib.machinery
import importlib

# Ensure the real websockets library is used instead of any test stub
if "websockets" in sys.modules:
    del sys.modules["websockets"]
import websockets

# Stub optional dependencies if missing
if importlib.util.find_spec("solders") is None:
    mod = types.ModuleType("solders")
    mod.__spec__ = importlib.machinery.ModuleSpec("solders", None)
    sys.modules.setdefault("solders", mod)
    sys.modules.setdefault("solders.keypair", types.SimpleNamespace(Keypair=type("Keypair", (), {})))
    sys.modules.setdefault("solders.pubkey", types.SimpleNamespace(Pubkey=object))
    sys.modules.setdefault("solders.instruction", types.SimpleNamespace(Instruction=object))
if importlib.util.find_spec("aiofiles") is None:
    aiof = types.ModuleType("aiofiles")
    aiof.__spec__ = importlib.machinery.ModuleSpec("aiofiles", None)
    sys.modules.setdefault("aiofiles", aiof)

from solhunter_zero import investor_demo
from solhunter_zero.event_bus import subscribe


@pytest.mark.asyncio
async def test_investor_demo_price_streams(tmp_path):
    events = []
    unsub = subscribe("price_update", lambda p: events.append(p))
    investor_demo.Memory = None

    async def orca_handler(ws):
        await ws.send(json.dumps({"token": "TOK", "price": 1.0}))
        await asyncio.sleep(0.1)

    async def ray_handler(ws):
        await ws.send(json.dumps({"token": "TOK", "price": 1.2}))
        await asyncio.sleep(0.1)

    server1 = await websockets.serve(orca_handler, "localhost", 0)
    port1 = server1.sockets[0].getsockname()[1]
    server2 = await websockets.serve(ray_handler, "localhost", 0)
    port2 = server2.sockets[0].getsockname()[1]

    await asyncio.to_thread(
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

    await asyncio.sleep(0.1)

    assert any(e.get("venue") == "orca" for e in events)
    assert any(e.get("venue") == "raydium" for e in events)
    assert not server1.connections
    assert not server2.connections

    unsub()
    server1.close()
    await server1.wait_closed()
    server2.close()
    await server2.wait_closed()
