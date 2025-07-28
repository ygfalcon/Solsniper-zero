import asyncio
import json
import sys
import types
import importlib.util
import websockets

# Stub heavy optional dependencies to keep import lightweight
dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
if importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
    sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
if importlib.util.find_spec("numpy") is None:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
if importlib.util.find_spec("aiofiles") is None:
    sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))
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
    sys.modules.setdefault("solders", types.ModuleType("solders"))
    sys.modules["solders.keypair"] = types.SimpleNamespace(Keypair=type("Keypair", (), {}))
    sys.modules["solders.pubkey"] = types.SimpleNamespace(Pubkey=object)
    sys.modules["solders.hash"] = types.SimpleNamespace(Hash=object)
    sys.modules["solders.message"] = types.SimpleNamespace(MessageV0=object)
    sys.modules["solders.transaction"] = types.SimpleNamespace(VersionedTransaction=object)
if importlib.util.find_spec("solana") is None:
    sys.modules.setdefault("solana", types.ModuleType("solana"))
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

    async with websockets.connect(f"ws://localhost:{port}") as ws:
        publish("ws", {"foo": 1})
        raw = await asyncio.wait_for(ws.recv(), timeout=1)
        data = json.loads(raw)
        assert data["topic"] == "ws"
        assert data["payload"] == {"foo": 1}
    await stop_ws_server()


@pytest.mark.asyncio
async def test_websocket_client_publish(monkeypatch):
    port = 8769
    await start_ws_server("localhost", port)
    received = []
    subscribe("remote", lambda p: received.append(p))
    await connect_ws(f"ws://localhost:{port}")
    await broadcast_ws(
        json.dumps({"topic": "remote", "payload": {"x": 5}}), to_clients=False
    )
    await asyncio.sleep(0.1)
    assert received == [{"x": 5}]
    await disconnect_ws()
    await stop_ws_server()


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

    monkeypatch.setenv("EVENT_BUS_URL", "ws://bus")
    monkeypatch.setattr(websockets, "connect", fake_connect)

    ev = importlib.reload(ev)
    await asyncio.sleep(0)

    assert called.get("url") == "ws://bus"

    await ev.disconnect_ws()
    monkeypatch.delenv("EVENT_BUS_URL", raising=False)
    importlib.reload(ev)


def test_publish_invalid_payload():
    from solhunter_zero.event_bus import publish
    with pytest.raises(ValueError):
        publish("action_executed", {"bad": 1})

