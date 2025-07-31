import asyncio
import types
import importlib.util
import importlib.machinery
import sys
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch_mod.__path__ = []
    torch_mod.Tensor = object
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch_mod.load = lambda *a, **k: {}
    class _Dev:
        def __init__(self, *a, **k):
            self.type = "cpu"
    torch_mod.device = _Dev
    torch_mod.tensor = lambda *a, **k: [0.0]
    torch_mod.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self,*exc: None)
    torch_mod.softmax = lambda x, dim=0: [1.0 / len(x) for _ in x]
    sys.modules.setdefault("torch", torch_mod)
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__spec__ = importlib.machinery.ModuleSpec("torch.nn", None)
    torch_nn.__path__ = []
    torch_nn.Module = type("Module", (), {"to": lambda self, *a, **k: None, "load_state_dict": lambda self, *a, **k: None})
    torch_nn.Sequential = lambda *a, **k: types.SimpleNamespace(__call__=lambda self,x:x, to=lambda self,*a,**k: None, eval=lambda self: None, load_state_dict=lambda *a, **k: None)
    torch_nn.Linear = lambda *a, **k: object()
    torch_nn.ReLU = lambda *a, **k: object()
    sys.modules.setdefault("torch.nn", torch_nn)
    torch_opt = types.ModuleType("torch.optim")
    torch_opt.__spec__ = importlib.machinery.ModuleSpec("torch.optim", None)
    torch_opt.__path__ = []
    sys.modules.setdefault("torch.optim", torch_opt)
    tud = types.ModuleType("torch.utils.data")
    tud.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", None)
    tud.Dataset = object
    tud.DataLoader = object
    sys.modules.setdefault("torch.utils.data", tud)
if importlib.util.find_spec("numpy") is None:
    np_mod = types.ModuleType("numpy")
    class _Array(list):
        @property
        def size(self):
            return len(self)
        def std(self):
            return 0.0
        def mean(self):
            return 0.0
    np_mod.asarray = lambda x, dtype=None: _Array(list(x))
    np_mod.mean = lambda x: 0.0
    np_mod.std = lambda x: 0.0
    sys.modules.setdefault("numpy", np_mod)
if ("aiohttp" not in sys.modules) and importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
if ("aiofiles" not in sys.modules) and importlib.util.find_spec("aiofiles") is None:
    aiof_mod = types.ModuleType("aiofiles")
    aiof_mod.__spec__ = importlib.machinery.ModuleSpec("aiofiles", None)
    sys.modules.setdefault("aiofiles", aiof_mod)
if importlib.util.find_spec("psutil") is None:
    sys.modules.setdefault("psutil", types.ModuleType("psutil"))
if importlib.util.find_spec("sqlalchemy") is None:
    sa = types.ModuleType("sqlalchemy")
    sa.create_engine = lambda *a, **k: None
    sa.MetaData = type("Meta", (), {"create_all": lambda *a, **k: None})
    sa.Column = lambda *a, **k: None
    sa.String = sa.Integer = sa.Float = sa.Numeric = sa.Text = object
    sa.DateTime = object
    sa.LargeBinary = object
    sa.ForeignKey = lambda *a, **k: None
    sa.select = lambda *a, **k: None
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
    async_mod = types.ModuleType("sqlalchemy.ext.asyncio")
    async_mod.create_async_engine = lambda *a, **k: None
    async_mod.async_sessionmaker = lambda *a, **k: lambda **kw: DummySession()
    async_mod.AsyncSession = object
    sys.modules.setdefault("sqlalchemy.ext.asyncio", async_mod)
if importlib.util.find_spec("sklearn") is None:
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules["sklearn.linear_model"] = types.SimpleNamespace(LinearRegression=object)
    sys.modules["sklearn.ensemble"] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
    sys.modules["sklearn.cluster"] = types.SimpleNamespace(KMeans=object, DBSCAN=object)
    sys.modules["sklearn.gaussian_process"] = types.SimpleNamespace(GaussianProcessRegressor=object)
    sys.modules["sklearn.gaussian_process.kernels"] = types.SimpleNamespace(Matern=object, RBF=object, ConstantKernel=object, C=object)
if importlib.util.find_spec("cachetools") is None:
    ct_mod = types.ModuleType("cachetools")
    class _Cache:
        def __init__(self, *a, **k):
            pass
    ct_mod.LRUCache = _Cache
    ct_mod.TTLCache = _Cache
    sys.modules.setdefault("cachetools", ct_mod)
if importlib.util.find_spec("transformers") is None:
    dummy_trans = types.ModuleType("transformers")
    dummy_trans.pipeline = lambda *a, **k: lambda x: []
    dummy_trans.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    sys.modules.setdefault("sentence_transformers", st_mod)
if importlib.util.find_spec("faiss") is None:
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
    sys.modules.setdefault("faiss", faiss_mod)
if "solhunter_zero.rl_daemon" not in sys.modules:
    rl_stub = types.ModuleType("solhunter_zero.rl_daemon")
    rl_stub.portfolio_state = lambda *a, **k: [0.0] * 8
    sys.modules["solhunter_zero.rl_daemon"] = rl_stub
if importlib.util.find_spec("watchfiles") is None:
    wf = types.ModuleType("watchfiles")
    wf.awatch = lambda *a, **k: []
    sys.modules.setdefault("watchfiles", wf)
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
event_pb2 = types.ModuleType("event_pb2")
for name in [
    "ActionExecuted",
    "WeightsUpdated",
    "RLWeights",
    "RLCheckpoint",
    "PortfolioUpdated",
    "ConfigUpdated",
    "DepthUpdate",
    "DepthServiceStatus",
    "Heartbeat",
    "TradeLogged",
    "RLMetrics",
    "PriceUpdate",
    "SystemMetrics",
    "ConfigUpdated",
    "PendingSwap",
    "RemoteSystemMetrics",
    "RiskMetrics",
    "RiskUpdated",
    "SystemMetricsCombined",
    "TokenDiscovered",
    "MemorySyncRequest",
    "MemorySyncResponse",
    "Event",
]:
    setattr(event_pb2, name, object())
sys.modules.setdefault("solhunter_zero.event_pb2", event_pb2)
try:
    import google
except ModuleNotFoundError:
    google = types.ModuleType("google")
    google.__path__ = []
    sys.modules.setdefault("google", google)
if importlib.util.find_spec("google.protobuf") is None:
    protobuf = types.ModuleType("protobuf")
    descriptor = types.ModuleType("descriptor")
    descriptor_pool = types.ModuleType("descriptor_pool")
    symbol_database = types.ModuleType("symbol_database")
    symbol_database.Default = lambda: object()
    internal = types.ModuleType("internal")
    internal.builder = types.ModuleType("builder")
    protobuf.descriptor = descriptor
    protobuf.descriptor_pool = descriptor_pool
    protobuf.symbol_database = symbol_database
    protobuf.internal = internal
    google.protobuf = protobuf
    sys.modules.setdefault("google.protobuf", protobuf)
    sys.modules.setdefault("google.protobuf.descriptor", descriptor)
    sys.modules.setdefault("google.protobuf.descriptor_pool", descriptor_pool)
    sys.modules.setdefault("google.protobuf.symbol_database", symbol_database)
    sys.modules.setdefault("google.protobuf.internal", internal)
    sys.modules.setdefault("google.protobuf.internal.builder", internal.builder)

import solhunter_zero.agents.hierarchical_rl as hr
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agents.swarm import AgentSwarm


class DummyPortfolio:
    def __init__(self):
        self.price_history = {}
        self.balances = {}


class DummyAgent:
    def __init__(self, name):
        self.name = name

    async def propose_trade(self, token, portfolio, *, depth=None, imbalance=None):
        return []


def test_hierarchical_policy_loading(monkeypatch, tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"x")
    called = {}

    def fake_load(p, *a, **k):
        called["path"] = str(p)
        return {"state": {}, "agents": ["a1", "a2"]}

    torch_stub = types.SimpleNamespace(
        tensor=lambda *a, **k: [],
        softmax=lambda x, dim=0: [0.5, 0.5],
        no_grad=lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self,*exc: None),
        load=fake_load,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    nn_stub = types.SimpleNamespace(
        Sequential=lambda *a, **k: types.SimpleNamespace(__call__=lambda self,x:[[0,0]],to=lambda self,*a,**k: None, eval=lambda self: None, load_state_dict=lambda *a, **k: None),
        Linear=lambda *a, **k: object(),
        ReLU=lambda *a, **k: object(),
    )
    monkeypatch.setattr(hr, "torch", torch_stub)
    monkeypatch.setattr(hr, "nn", nn_stub)

    agent = hr.HierarchicalRLAgent(["a1", "a2"], model_path=path)
    assert called["path"] == str(path)


def test_agent_manager_uses_hierarchical_weights(monkeypatch):
    class DummyHier(hr.HierarchicalRLAgent):
        def __init__(self):
            self.agent_names = ["a1", "a2"]
        def compute_weights(self, portfolio, token, price, *, depth=0.0, tx_rate=0.0):
            return {"a1": 2.0, "a2": 0.5}
        async def propose_trade(self, token, portfolio, *, depth=None, imbalance=None):
            return []

    hier = DummyHier()
    agents = [DummyAgent("a1"), DummyAgent("a2"), hier]
    class DummyMemory:
        def list_trades(self, *a, **k):
            return []

    mgr = AgentManager(agents, memory_agent=MemoryAgent(DummyMemory()))
    pf = DummyPortfolio()
    pf.price_history["tok"] = [1.0]
    record = {}
    async def fake_propose(self, token, portfolio, *, weights=None, rl_action=None):
        record["weights"] = dict(weights)
        return []
    monkeypatch.setattr(AgentSwarm, "propose", fake_propose, raising=False)
    asyncio.run(mgr.evaluate("tok", pf))
    assert record["weights"]["a1"] == 2.0
    assert record["weights"]["a2"] == 0.5
