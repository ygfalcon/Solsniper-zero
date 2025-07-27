import asyncio
import json
import os
import subprocess
import sys

import pytest
import websockets
from aiohttp import web


@pytest.mark.asyncio
async def test_depth_service_event_bus(tmp_path):
    # Build the service
    subprocess.run([
        "cargo",
        "build",
        "--manifest-path",
        "depth_service/Cargo.toml",
    ], check=True)

    # Start dummy RPC server
    async def rpc_handler(request):
        data = await request.json()
        method = data.get("method")
        if method == "getLatestBlockhash":
            return web.json_response({
                "jsonrpc": "2.0",
                "result": {
                    "context": {"slot": 1},
                    "value": {
                        "blockhash": "11111111111111111111111111111111",
                        "lastValidBlockHeight": 1,
                    },
                },
                "id": data.get("id"),
            })
        elif method == "getVersion":
            return web.json_response({
                "jsonrpc": "2.0",
                "result": {"solana-core": "1.18.0"},
                "id": data.get("id"),
            })
        return web.json_response({"jsonrpc": "2.0", "result": None, "id": data.get("id")})

    rpc_app = web.Application()
    rpc_app.router.add_post("/", rpc_handler)
    rpc_runner = web.AppRunner(rpc_app)
    await rpc_runner.setup()
    rpc_site = web.TCPSite(rpc_runner, "localhost", 0)
    await rpc_site.start()
    rpc_port = rpc_site._server.sockets[0].getsockname()[1]

    # Start event bus server
    events = []

    async def bus_handler(ws):
        async for msg in ws:
            try:
                events.append(json.loads(msg))
            except Exception:
                pass

    bus_server = await websockets.serve(bus_handler, "localhost", 0)
    bus_port = bus_server.sockets[0].getsockname()[1]

    # Start feed server producing a single update
    async def feed_handler(ws):
        await ws.send(json.dumps({"token": "TOK", "bids": 1, "asks": 2}))
        await asyncio.sleep(0.2)

    feed_server = await websockets.serve(feed_handler, "localhost", 0)
    feed_port = feed_server.sockets[0].getsockname()[1]

    env = os.environ.copy()
    env.update({
        "EVENT_BUS_URL": f"ws://localhost:{bus_port}",
        "SOLANA_RPC_URL": f"http://localhost:{rpc_port}",
    })

    proc = await asyncio.create_subprocess_exec(
        "depth_service/target/debug/depth_service",
        "--serum",
        f"ws://localhost:{feed_port}",
        env=env,
    )

    # Wait for message
    for _ in range(50):
        if events:
            break
        await asyncio.sleep(0.1)
    proc.kill()
    await proc.wait()

    await rpc_runner.cleanup()
    bus_server.close()
    await bus_server.wait_closed()
    feed_server.close()
    await feed_server.wait_closed()

    assert events
    assert events[0]["topic"] == "depth_update"
    assert "TOK" in events[0]["payload"]

