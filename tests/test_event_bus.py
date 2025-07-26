import asyncio
import json
import sys
import types

# Stub heavy optional dependencies to keep import lightweight
dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
sys.modules.setdefault("transformers", dummy_trans)
sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))
sys.modules.setdefault("solders", types.ModuleType("solders"))
sys.modules["solders.keypair"] = types.SimpleNamespace(Keypair=type("Keypair", (), {}))
sys.modules["solders.pubkey"] = types.SimpleNamespace(Pubkey=object)
sys.modules["solders.hash"] = types.SimpleNamespace(Hash=object)
sys.modules["solders.message"] = types.SimpleNamespace(MessageV0=object)
sys.modules["solders.transaction"] = types.SimpleNamespace(VersionedTransaction=object)
sys.modules.setdefault("solana", types.ModuleType("solana"))
sys.modules["solana.rpc"] = types.ModuleType("rpc")
sys.modules["solana.rpc.api"] = types.SimpleNamespace(Client=object)
sys.modules["solana.rpc.async_api"] = types.SimpleNamespace(AsyncClient=object)
sys.modules["solana.rpc.websocket_api"] = types.SimpleNamespace(connect=lambda *a, **k: None)
sys.modules["solana.rpc.websocket_api"].RpcTransactionLogsFilterMentions = object

import pytest

from solhunter_zero.event_bus import subscribe, publish
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.memory import Memory
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
    mem = Memory("sqlite:///:memory:")
    mem_agent = MemoryAgent(mem)

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

    assert received and received[0]["action"]["token"] == "TOK"

