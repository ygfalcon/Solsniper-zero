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

