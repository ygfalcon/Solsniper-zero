import asyncio
import sys
import types
import importlib.machinery


def _prepare(monkeypatch):
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    st_mod.SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
        get_sentence_embedding_dimension=lambda: 1,
        encode=lambda x: [0],
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)

    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(
        sys.modules,
        "sklearn.linear_model",
        types.SimpleNamespace(LinearRegression=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.ensemble",
        types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.cluster",
        types.SimpleNamespace(KMeans=object, DBSCAN=object),
    )

    xgb_mod = types.ModuleType("xgboost")
    xgb_mod.__spec__ = importlib.machinery.ModuleSpec("xgboost", None)
    xgb_mod.XGBRegressor = object
    monkeypatch.setitem(sys.modules, "xgboost", xgb_mod)

    models_mod = types.ModuleType("solhunter_zero.models")
    models_mod.get_model = lambda *a, **k: None
    models_mod.load_compiled_model = lambda *a, **k: None
    models_mod.export_torchscript = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "solhunter_zero.models", models_mod)
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.models.regime_classifier",
        types.SimpleNamespace(get_model=lambda *a, **k: None),
    )
    models_mod.regime_classifier = sys.modules[
        "solhunter_zero.models.regime_classifier"
    ]

    memory_mod = types.ModuleType("solhunter_zero.memory")

    class DummyMemory:
        def __init__(self, url: str = "sqlite:///:memory:"):
            self.trades = []

        async def log_trade(self, **kw):
            self.trades.append(types.SimpleNamespace(**kw))

        def list_trades(self, limit: int | None = None):
            return self.trades[:limit] if limit is not None else self.trades

        def log_var(self, value: float) -> None:
            pass

        def list_vars(self):
            return []

    memory_mod.Memory = DummyMemory
    monkeypatch.setitem(sys.modules, "solhunter_zero.memory", memory_mod)

    portfolio_mod = types.ModuleType("solhunter_zero.portfolio")

    class DummyPortfolio:
        def __init__(self, path=None):
            self.path = path
            self.balances = {}
            self.risk_metrics = {}
            self.price_history = {}

        async def update_async(self, token, amount, price):
            pass

        def total_value(self, prices=None):
            return 0.0

        def percent_allocated(self, token, prices=None):
            return 0.0

        def update_drawdown(self, prices):
            pass

        def record_prices(self, prices):
            pass

        def update_risk_metrics(self):
            pass

        def current_drawdown(self, prices=None):
            return 0.0

        async def update_highs_async(self, prices):
            pass

        def trailing_stop_triggered(self, token, price, ts):
            return False

    portfolio_mod.Portfolio = DummyPortfolio
    portfolio_mod.calculate_order_size = lambda *a, **k: 1.0
    portfolio_mod.dynamic_order_size = lambda *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "solhunter_zero.portfolio", portfolio_mod)

    agent_manager_mod = types.ModuleType("solhunter_zero.agent_manager")

    class DummyAgentManager:
        def __init__(self, *a, **k):
            pass

        async def execute(self, token, portfolio):
            pass

        def evolve(self, threshold: float = 0.0):
            pass

    agent_manager_mod.AgentManager = DummyAgentManager
    monkeypatch.setitem(sys.modules, "solhunter_zero.agent_manager", agent_manager_mod)

    adv_mod = types.ModuleType("solhunter_zero.advanced_memory")
    adv_mod.AdvancedMemory = object
    monkeypatch.setitem(sys.modules, "solhunter_zero.advanced_memory", adv_mod)

    discovery_mod = types.ModuleType("solhunter_zero.agents.discovery")

    class StubDiscoveryAgent:
        async def discover_tokens(self, **kwargs):
            return ["TOK"]

    discovery_mod.DiscoveryAgent = StubDiscoveryAgent
    monkeypatch.setitem(sys.modules, "solhunter_zero.agents.discovery", discovery_mod)


async def _run_script(iterations=100):
    from scripts.paper_test import run_paper_test

    mem, roi = await run_paper_test(iterations=iterations)
    return mem, roi


def test_paper_extended(monkeypatch):
    _prepare(monkeypatch)
    mem, roi = asyncio.run(_run_script(iterations=100))
    assert len(mem.list_trades()) >= 100
    assert all(isinstance(v, float) for v in roi.values())
