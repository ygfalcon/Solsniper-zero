import types
import sys
import importlib.util
import numpy as np

# Stub torch if missing
if importlib.util.find_spec("torch") is None:
    dummy = types.ModuleType("torch")
    dummy.float32 = "float32"
    def _tensor(x, dtype=None):
        return np.array(x)
    class _Ctx:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False
    dummy.tensor = _tensor
    dummy.no_grad = lambda: _Ctx()
    dummy.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    sys.modules.setdefault("torch", dummy)
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
if "google" not in sys.modules:
    sys.modules["google"] = types.ModuleType("google")
    pb = types.ModuleType("google.protobuf")
    pb.descriptor = object()
    pb.descriptor_pool = object()
    pb.runtime_version = "0"
    pb.symbol_database = object()
    sys.modules["google.protobuf"] = pb

import pytest
try:
    from solhunter_zero.graph_swarm import GraphSwarm
    from solhunter_zero.agents.memory import MemoryAgent
    from solhunter_zero.advanced_memory import AdvancedMemory
except Exception:  # pragma: no cover - optional deps missing
    pytest.skip("heavy dependencies missing", allow_module_level=True)

class DummyAgent:
    def __init__(self, name):
        self.name = name

async def dummy():
    return []
DummyAgent.propose_trade = lambda *a, **k: []


def test_graph_swarm_weights():
    mem = AdvancedMemory(url="sqlite:///:memory:")
    mem.log_trade(token="T", direction="buy", amount=1, price=1, reason="a1")
    mem.log_trade(token="T", direction="sell", amount=1, price=1.1, reason="a1")
    mem.log_trade(token="T", direction="buy", amount=1, price=1, reason="a2")
    mem.log_trade(token="T", direction="sell", amount=1, price=0.9, reason="a2")
    mem_agent = MemoryAgent(mem)
    gs = GraphSwarm(mem_agent, base_weights={"a1": 1.0, "a2": 1.0})
    gs.model = types.SimpleNamespace(__call__=lambda x, y: types.SimpleNamespace(numpy=lambda: np.array([0.3, 0.7])))
    gs._roi_by_agent = lambda names: {n: 0.0 for n in names}
    agents = [DummyAgent("a1"), DummyAgent("a2")]
    w = gs.compute_weights(agents)
    assert abs(w["a1"] - 0.3) < 1e-6
    assert abs(w["a2"] - 0.7) < 1e-6

