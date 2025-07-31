import asyncio
import json
import sys
import types
import importlib.util
import websockets

# Stub heavy optional dependencies to keep import lightweight
dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
import importlib.machinery
dummy_trans.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    sys.modules.setdefault("sentence_transformers", st_mod)
if importlib.util.find_spec("faiss") is None:
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
    sys.modules.setdefault("faiss", faiss_mod)
if importlib.util.find_spec("torch") is None:
    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch_mod.__path__ = []
    torch_mod.Tensor = object
    sys.modules.setdefault("torch", torch_mod)
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__spec__ = importlib.machinery.ModuleSpec("torch.nn", None)
    torch_nn.__path__ = []
    torch_nn.Module = type("Module", (), {})
    sys.modules.setdefault("torch.nn", torch_nn)
    torch_opt = types.ModuleType("torch.optim")
    torch_opt.__spec__ = importlib.machinery.ModuleSpec("torch.optim", None)
    torch_opt.__path__ = []
    sys.modules.setdefault("torch.optim", torch_opt)
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.__spec__ = importlib.machinery.ModuleSpec("torch.utils", None)
    torch_utils.__path__ = []
    sys.modules.setdefault("torch.utils", torch_utils)
    tud = types.ModuleType("torch.utils.data")
    tud.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", None)
    tud.Dataset = object
    tud.DataLoader = object
    sys.modules.setdefault("torch.utils.data", tud)
if importlib.util.find_spec("numpy") is None:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
if importlib.util.find_spec("aiofiles") is None:
    aiof_mod = types.ModuleType("aiofiles")
    aiof_mod.__spec__ = importlib.machinery.ModuleSpec("aiofiles", None)
    sys.modules.setdefault("aiofiles", aiof_mod)
if importlib.util.find_spec("sqlalchemy") is None:
    sa = types.ModuleType("sqlalchemy")
    sa.create_engine = lambda *a, **k: None
    sa.MetaData = type("Meta", (), {"create_all": lambda *a, **k: None})
    sa.Column = lambda *a, **k: None
    sa.String = sa.Integer = sa.Float = sa.Numeric = sa.Text = object
    sa.DateTime = object
    sa.ForeignKey = lambda *a, **k: None
    sys.modules.setdefault("sqlalchemy", sa)
    orm = types.ModuleType("orm")

    def declarative_base(*a, **k):
        return type("Base", (), {"metadata": sa.MetaData()})

    orm.declarative_base = declarative_base

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def add(self, *a, **k):
            pass

        def commit(self):
            pass

        def query(self, *a, **k):
            return []

    orm.sessionmaker = lambda *a, **k: lambda **kw: DummySession()
    sys.modules.setdefault("sqlalchemy.orm", orm)
if importlib.util.find_spec("solders") is None:
    s_mod = types.ModuleType("solders")
    s_mod.__spec__ = importlib.machinery.ModuleSpec("solders", None)
    sys.modules.setdefault("solders", s_mod)
    sys.modules["solders.keypair"] = types.SimpleNamespace(Keypair=type("Keypair", (), {}))
    sys.modules["solders.pubkey"] = types.SimpleNamespace(Pubkey=object)
    sys.modules["solders.hash"] = types.SimpleNamespace(Hash=object)
    sys.modules["solders.message"] = types.SimpleNamespace(MessageV0=object)
    sys.modules["solders.transaction"] = types.SimpleNamespace(VersionedTransaction=object)
if importlib.util.find_spec("solana") is None:
    sol_mod = types.ModuleType("solana")
    sol_mod.__spec__ = importlib.machinery.ModuleSpec("solana", None)
    sys.modules.setdefault("solana", sol_mod)
    sys.modules["solana.rpc"] = types.ModuleType("rpc")
    sys.modules["solana.rpc.api"] = types.SimpleNamespace(Client=object)
    sys.modules["solana.rpc.async_api"] = types.SimpleNamespace(AsyncClient=object)
    sys.modules["solana.rpc.websocket_api"] = types.SimpleNamespace(connect=lambda *a, **k: None)
    sys.modules["solana.rpc.websocket_api"].RpcTransactionLogsFilterMentions = object

import pytest

from solhunter_zero.event_bus import (
    subscribe,
    publish,
    subscription,
    start_ws_server,
    stop_ws_server,
    connect_ws,
    disconnect_ws,
    broadcast_ws,
    _unpack_batch,
    _maybe_decompress,
)
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.portfolio import Portfolio


@pytest.mark.asyncio
async def test_publish_subscribe_basic():
    events = []

    async def handler(payload):
        events.append(payload)

    unsub = subscribe("test", handler)
    publish("test", {"a": 1})
    await asyncio.sleep(0)
    assert events == [{"a": 1}]

    unsub()
    publish("test", {"b": 2})
    await asyncio.sleep(0)
    assert events == [{"a": 1}]


@pytest.mark.asyncio
async def test_agent_manager_emits_events(monkeypatch):
    class DummyMemory:
        def log_trade(self, **kw):
            pass

        def list_trades(self):
            return []

    mem_agent = MemoryAgent(DummyMemory())

    class DummyExec(ExecutionAgent):
        async def execute(self, action):
            return {"ok": True}

    mgr = AgentManager([], executor=DummyExec(), memory_agent=mem_agent)

    async def fake_evaluate(self, token, pf):
        return [{"token": token, "side": "buy", "amount": 1.0, "price": 1.0}]

    monkeypatch.setattr(AgentManager, "evaluate", fake_evaluate)

    received = []

    async def on_action(payload):
        received.append(payload)

    subscribe("action_executed", on_action)

    await mgr.execute("TOK", Portfolio(path=None))
    await asyncio.sleep(0)

    from solhunter_zero.schemas import ActionExecuted

    assert received and isinstance(received[0], ActionExecuted)
    assert received[0].action["token"] == "TOK"


@pytest.mark.asyncio
async def test_subscription_context_manager():
    seen = []

    async def handler(payload):
        seen.append(payload)

    with subscription("ctx", handler):
        publish("ctx", {"msg": 1})
        await asyncio.sleep(0)

    publish("ctx", {"msg": 2})
    await asyncio.sleep(0)

    assert seen == [{"msg": 1}]


@pytest.mark.asyncio
async def test_websocket_publish_and_receive():
    port = 8768
    await start_ws_server("localhost", port)

    from solhunter_zero import event_pb2
    async with websockets.connect(f"ws://localhost:{port}") as ws:
        publish("weights_updated", {"weights": {"x": 1.0}})
        raw = await asyncio.wait_for(ws.recv(), timeout=1)
        if isinstance(raw, bytes):
            ev = event_pb2.Event()
            ev.ParseFromString(raw)
            assert ev.topic == "weights_updated"
            assert dict(ev.weights_updated.weights)["x"] == 1.0
        else:
            raise AssertionError("expected binary message")
    await stop_ws_server()


@pytest.mark.asyncio
async def test_websocket_client_publish(monkeypatch):
    port = 8769
    await start_ws_server("localhost", port)
    received = []
    subscribe("weights_updated", lambda p: received.append(p))
    await connect_ws(f"ws://localhost:{port}")
    from solhunter_zero import event_pb2
    upd = event_pb2.WeightsUpdated(weights={"x": 5.0})
    ev = event_pb2.Event(topic="weights_updated", weights_updated=upd)
    await broadcast_ws(ev.SerializeToString(), to_clients=False)
    await asyncio.sleep(0.1)
    from solhunter_zero.schemas import WeightsUpdated
    assert received and isinstance(received[0], WeightsUpdated)
    assert received[0].weights["x"] == 5.0
    await disconnect_ws()
    await stop_ws_server()


@pytest.mark.asyncio
async def test_websocket_reconnect_on_drop():
    port = 8770
    connections = []

    async def handler(ws):
        connections.append(ws)
        if len(connections) == 1:
            await asyncio.sleep(0.05)
            await ws.close()
        else:
            await ws.send(json.dumps({"topic": "weights_updated", "payload": {"weights": {"x": 7}}}))
            await asyncio.sleep(0.1)

    server = await websockets.serve(handler, "localhost", port)

    received = []
    subscribe("weights_updated", lambda p: received.append(p))
    await connect_ws(f"ws://localhost:{port}")

    for _ in range(50):
        if len(connections) >= 2:
            break
        await asyncio.sleep(0.1)

    await asyncio.sleep(0.2)

    await disconnect_ws()
    server.close()
    await server.wait_closed()

    from solhunter_zero.schemas import WeightsUpdated

    assert len(connections) >= 2
    assert received and isinstance(received[-1], WeightsUpdated)
    assert received[-1].weights["x"] == 7


@pytest.mark.asyncio
async def test_event_bus_url_connect(monkeypatch):
    import importlib
    import solhunter_zero.event_bus as ev

    called = {}

    async def fake_connect(url):
        called["url"] = url

        class Dummy:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def send(self, _):
                pass

            async def close(self):
                pass

        return Dummy()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    monkeypatch.setattr(
        "solhunter_zero.config.get_event_bus_url", lambda *_: "ws://bus"
    )

    ev = importlib.reload(ev)
    ev._reload_bus(None)
    await asyncio.sleep(0)

    assert called.get("url") == "ws://bus"

    await ev.disconnect_ws()
    importlib.reload(ev)


@pytest.mark.asyncio
async def test_multiple_peer_connections():
    port1 = 8791
    port2 = 8792

    recv1 = []
    recv2 = []

    async def h1(ws):
        async for msg in ws:
            recv1.append(msg)

    async def h2(ws):
        async for msg in ws:
            recv2.append(msg)

    server1 = await websockets.serve(h1, "localhost", port1)
    server2 = await websockets.serve(h2, "localhost", port2)

    await connect_ws(f"ws://localhost:{port1}")
    await connect_ws(f"ws://localhost:{port2}")
    from solhunter_zero import event_pb2
    upd = event_pb2.WeightsUpdated(weights={"x": 9.0})
    ev = event_pb2.Event(topic="weights_updated", weights_updated=upd)
    await broadcast_ws(ev.SerializeToString(), to_clients=False)
    await asyncio.sleep(0.1)

    assert recv1 and recv2

    await disconnect_ws()
    server1.close()
    server2.close()
    await server1.wait_closed()
    await server2.wait_closed()


def test_event_bus_fallback_json(monkeypatch):
    import builtins
    import importlib
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "orjson":
            raise ModuleNotFoundError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import solhunter_zero.event_bus as ev
    ev = importlib.reload(ev)

    seen = []
    ev.subscribe("x", lambda p: seen.append(p))
    ev.publish("x", {"a": 1})

    assert seen == [{"a": 1}]
    assert not ev._USE_ORJSON

    importlib.reload(ev)


def test_publish_invalid_payload():
    from solhunter_zero.event_bus import publish
    with pytest.raises(ValueError):
        publish("action_executed", {"bad": 1})


def test_event_compression_algorithms(monkeypatch):
    import importlib
    from solhunter_zero import event_pb2
    import solhunter_zero.event_bus as ev

    monkeypatch.setenv("EVENT_COMPRESSION", "lz4")
    ev = importlib.reload(ev)
    data = ev._encode_event("weights_updated", event_pb2.WeightsUpdated(weights={"x": 1.0}))
    ev_msg = event_pb2.Event()
    ev_msg.ParseFromString(ev._maybe_decompress(data))
    assert ev_msg.topic == "weights_updated"

    monkeypatch.setenv("EVENT_COMPRESSION", "zstd")
    ev = importlib.reload(ev)
    data = ev._encode_event("weights_updated", event_pb2.WeightsUpdated(weights={"x": 1.0}))
    ev_msg.ParseFromString(ev._maybe_decompress(data))
    assert ev_msg.topic == "weights_updated"

    monkeypatch.setenv("EVENT_COMPRESSION", "zlib")
    ev = importlib.reload(ev)
    data = ev._encode_event("weights_updated", event_pb2.WeightsUpdated(weights={"x": 1.0}))
    ev_msg.ParseFromString(ev._maybe_decompress(data))
    assert ev_msg.topic == "weights_updated"

    monkeypatch.setenv("EVENT_COMPRESSION", "none")
    ev = importlib.reload(ev)
    data = ev._encode_event("weights_updated", event_pb2.WeightsUpdated(weights={"x": 1.0}))
    ev_msg.ParseFromString(data)
    assert ev_msg.topic == "weights_updated"


@pytest.mark.asyncio
async def test_zstd_round_trip(monkeypatch):
    import importlib
    monkeypatch.setenv("COMPRESS_EVENTS", "1")
    monkeypatch.setenv("EVENT_COMPRESSION_THRESHOLD", "0")
    monkeypatch.delenv("EVENT_COMPRESSION", raising=False)
    import solhunter_zero.event_bus as ev
    ev = importlib.reload(ev)
    if not ev._HAS_ZSTD:
        pytest.skip("zstandard not installed")

    port = 8780
    await ev.start_ws_server("localhost", port)
    from solhunter_zero import event_pb2
    async with websockets.connect(f"ws://localhost:{port}") as ws:
        ev.publish("depth_service_status", {"status": "ok"})
        raw = await asyncio.wait_for(ws.recv(), timeout=1)
        assert isinstance(raw, bytes)
        assert raw.startswith(b"(\xb5/\xfd")
        ev_msg = event_pb2.Event()
        ev_msg.ParseFromString(ev._maybe_decompress(raw))
        assert ev_msg.depth_service_status.status == "ok"
    await ev.stop_ws_server()


@pytest.mark.asyncio
async def test_websocket_batching():
    import solhunter_zero.event_bus as ev
    port = 8793
    await start_ws_server("localhost", port)
    from solhunter_zero import event_pb2
    async with websockets.connect(f"ws://localhost:{port}") as ws:
        ev.publish("weights_updated", {"weights": {"x": 1}})
        ev.publish("weights_updated", {"weights": {"x": 2}})
        raw = await asyncio.wait_for(ws.recv(), timeout=1)
        msgs = _unpack_batch(raw)
        assert msgs is not None and len(msgs) == 2
        values = []
        for m in msgs:
            ev_msg = event_pb2.Event()
            ev_msg.ParseFromString(_maybe_decompress(m))
            values.append(dict(ev_msg.weights_updated.weights)["x"])
        assert set(values) == {1.0, 2.0}
    await stop_ws_server()

