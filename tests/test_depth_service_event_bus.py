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
    topics = [e.get("topic") for e in events]
    assert "depth_update" in topics
    assert "depth_service_status" in topics
    first_update = next(e for e in events if e.get("topic") == "depth_update")
    assert "TOK" in first_update["payload"]


@pytest.mark.asyncio
async def test_depth_service_event_bus_reconnect(tmp_path):
    subprocess.run([
        "cargo",
        "build",
        "--manifest-path",
        "depth_service/Cargo.toml",
    ], check=True)

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

    events = []

    async def bus_handler(ws):
        async for msg in ws:
            try:
                events.append(json.loads(msg))
            except Exception:
                pass

    bus_server = await websockets.serve(bus_handler, "localhost", 0)
    bus_port = bus_server.sockets[0].getsockname()[1]

    async def feed_handler(ws):
        while True:
            await ws.send(json.dumps({"token": "TOK", "bids": 1, "asks": 2}))
            await asyncio.sleep(0.05)

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

    for _ in range(50):
        if any(e.get("topic") == "depth_update" for e in events):
            break
        await asyncio.sleep(0.1)

    initial_len = len(events)

    bus_server.close()
    await bus_server.wait_closed()

    await asyncio.sleep(0.5)

    bus_server = await websockets.serve(bus_handler, "localhost", bus_port)

    for _ in range(100):
        if len(events) > initial_len:
            break
        await asyncio.sleep(0.1)

    proc.kill()
    await proc.wait()

    await rpc_runner.cleanup()
    bus_server.close()
    await bus_server.wait_closed()
    feed_server.close()
    await feed_server.wait_closed()

    new_events = events[initial_len:]
    assert any(e.get("topic") == "depth_service_status" for e in new_events)
    assert len([e for e in events if e.get("topic") == "depth_update"]) > (
        len([e for e in events[:initial_len] if e.get("topic") == "depth_update"])
    )


@pytest.mark.asyncio
async def test_depth_service_route_search(tmp_path):
    subprocess.run([
        "cargo",
        "build",
        "--manifest-path",
        "depth_service/Cargo.toml",
    ], check=True)

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

    async def feed_handler_1(ws):
        await ws.send(json.dumps({"token": "TOK", "price": 10}))
        await asyncio.sleep(0.2)

    async def feed_handler_2(ws):
        await ws.send(json.dumps({"token": "TOK", "price": 12}))
        await asyncio.sleep(0.2)

    feed_server_1 = await websockets.serve(feed_handler_1, "localhost", 0)
    port1 = feed_server_1.sockets[0].getsockname()[1]
    feed_server_2 = await websockets.serve(feed_handler_2, "localhost", 0)
    port2 = feed_server_2.sockets[0].getsockname()[1]

    env = os.environ.copy()
    env.update({"SOLANA_RPC_URL": f"http://localhost:{rpc_port}"})

    proc = await asyncio.create_subprocess_exec(
        "depth_service/target/debug/depth_service",
        "--raydium",
        f"ws://localhost:{port1}",
        "--orca",
        f"ws://localhost:{port2}",
        env=env,
    )

    await asyncio.sleep(1.0)

    from solhunter_zero import depth_client

    res = await depth_client.best_route("TOK", 1.0, max_hops=4)

    proc.kill()
    await proc.wait()

    await rpc_runner.cleanup()
    feed_server_1.close()
    await feed_server_1.wait_closed()
    feed_server_2.close()
    await feed_server_2.wait_closed()

    assert res is not None
    path, profit, slip = res
    assert path == ["raydium", "orca"]
    assert profit == pytest.approx(0.1666, rel=1e-2)
    assert slip == pytest.approx(0.1833, rel=1e-2)

