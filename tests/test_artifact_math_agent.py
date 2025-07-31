import asyncio
import io
import json
import builtins
import pytest
import sys
import types

if "google.protobuf" not in sys.modules:
    google = types.ModuleType("google")
    protobuf = types.ModuleType("protobuf")
    google.protobuf = protobuf
    sys.modules.setdefault("google", google)
    sys.modules.setdefault("google.protobuf", protobuf)
dummy_pb = types.ModuleType("solhunter_zero.event_pb2")
for name in [
    "Event",
    "ActionExecuted",
    "WeightsUpdated",
    "RLWeights",
    "RLCheckpoint",
    "PortfolioUpdated",
    "DepthUpdate",
    "DepthServiceStatus",
    "Heartbeat",
    "TradeLogged",
    "RLMetrics",
    "SystemMetrics",
    "PriceUpdate",
    "ConfigUpdated",
    "PendingSwap",
    "RemoteSystemMetrics",
    "RiskMetrics",
    "RiskUpdated",
    "SystemMetricsCombined",
    "TokenDiscovered",
    "MemorySyncRequest",
    "MemorySyncResponse",
]:
    setattr(dummy_pb, name, type(name, (), {}))
sys.modules.setdefault("solhunter_zero.event_pb2", dummy_pb)

artifact_math = pytest.importorskip("solhunter_zero.agents.artifact_math_agent")

ArtifactMathAgent = artifact_math.ArtifactMathAgent
map_glyph_series = artifact_math.map_glyph_series
aggregate_scores = artifact_math.aggregate_scores
from solhunter_zero.datasets.artifact_math import _DEFAULT_PATH as DATA_PATH
from solhunter_zero.portfolio import Portfolio, Position

class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_map_glyph_series():
    mapping = {"A": 1, "B": -1}
    series = ["A", "B", "A"]
    assert map_glyph_series(series, mapping) == [1, -1, 1]


def test_aggregate_scores():
    assert aggregate_scores([1, 2, 3]) == 8


def test_propose_trade_buy(monkeypatch):
    monkeypatch.setattr(artifact_math, "map_glyph_series", lambda s, m: [1, 1])
    monkeypatch.setattr(artifact_math, "aggregate_scores", lambda scores: 1)
    monkeypatch.setattr(builtins, "open", lambda *a, **k: io.StringIO("{}"))
    monkeypatch.setattr(json, "load", lambda f: {"A": 1})
    agent = ArtifactMathAgent(threshold=0, dataset_path=DATA_PATH)
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("TOK", pf))
    assert actions and actions[0]["side"] == "buy"


def test_propose_trade_sell(monkeypatch):
    monkeypatch.setattr(artifact_math, "map_glyph_series", lambda s, m: [-1])
    monkeypatch.setattr(artifact_math, "aggregate_scores", lambda scores: -1)
    monkeypatch.setattr(builtins, "open", lambda *a, **k: io.StringIO("{}"))
    monkeypatch.setattr(json, "load", lambda f: {"A": 1})
    agent = ArtifactMathAgent(threshold=0, dataset_path=DATA_PATH)
    pf = DummyPortfolio()
    pf.balances["TOK"] = Position("TOK", 1, 1.0, 1.0)
    actions = asyncio.run(agent.propose_trade("TOK", pf))
    assert actions and actions[0]["side"] == "sell"
