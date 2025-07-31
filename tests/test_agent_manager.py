import json
import asyncio
import sys
import types
import importlib.util

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

from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory


def test_update_weights_persists(tmp_path):
    path = tmp_path / "w.json"
    mem = Memory("sqlite:///:memory:")
    mem_agent = MemoryAgent(mem)
    mgr = AgentManager([], memory_agent=mem_agent, weights={"a": 1.0}, weights_path=str(path))

    mem.log_trade(token="tok", direction="buy", amount=1, price=1, reason="a")
    mem.log_trade(token="tok", direction="sell", amount=1, price=2, reason="a")

    mgr.update_weights()

    assert path.exists()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data.get("a", 0) > 1.0


def test_rl_weights_event_updates_manager(tmp_path):
    path = tmp_path / "w.json"
    mem = Memory("sqlite:///:memory:")
    mem_agent = MemoryAgent(mem)
    mgr = AgentManager([], memory_agent=mem_agent, weights={}, weights_path=str(path))

    from solhunter_zero.event_bus import publish
    from solhunter_zero.schemas import RLWeights

    publish("rl_weights", RLWeights(weights={"b": 2.0}, risk={"risk_multiplier": 1.1}))
    asyncio.run(asyncio.sleep(0))

    assert mgr.weights.get("b") == 2.0
    assert path.exists()


def test_rotate_weight_configs_selects_best(tmp_path):
    mem = Memory("sqlite:///:memory:")
    mem.log_trade(token="tok", direction="buy", amount=1, price=1, reason="a1")
    mem.log_trade(token="tok", direction="sell", amount=1, price=2, reason="a1")
    mem.log_trade(token="tok", direction="buy", amount=1, price=1, reason="a2")
    mem.log_trade(token="tok", direction="sell", amount=1, price=0.5, reason="a2")
    mem_agent = MemoryAgent(mem)

    w1 = tmp_path / "w1.toml"
    w1.write_text("""[agent_weights]\na1 = 1.0\na2 = 1.0\n""")
    w2 = tmp_path / "w2.toml"
    w2.write_text("""[agent_weights]\na1 = 2.0\na2 = 0.5\n""")

    mgr = AgentManager(
        [],
        memory_agent=mem_agent,
        weights_path=str(tmp_path / "active.json"),
        weight_config_paths=[str(w1), str(w2)],
    )

    mgr.rotate_weight_configs()
    assert mgr.weights["a1"] == 2.0
    assert mgr.weights["a2"] == 0.5

