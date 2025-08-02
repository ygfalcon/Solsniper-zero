import asyncio
import importlib
import importlib.machinery
import sys
import types

from solhunter_zero.trade_analyzer import TradeAnalyzer


def _load_main(monkeypatch):
    """Import ``solhunter_zero.main`` with stubbed dependencies."""
    # Optional deps used during import
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

        def log_trade(self, **kw):
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

        def add(self, token, amount, price):
            self.update(token, amount, price)

        def update(self, token, amount, price):
            pos = self.balances.get(token)
            if pos is None:
                if amount > 0:
                    self.balances[token] = types.SimpleNamespace(
                        amount=amount, entry_price=price, high_price=price
                    )
            else:
                new_amount = pos.amount + amount
                if new_amount <= 0:
                    self.balances.pop(token, None)
                else:
                    pos.amount = new_amount

        async def update_async(self, token, amount, price):
            self.update(token, amount, price)

        def total_value(self, prices=None):
            total = 0.0
            for t, pos in self.balances.items():
                price = prices.get(t, 1.0) if prices else 1.0
                total += pos.amount * price
            return total

        def percent_allocated(self, token, prices=None):
            tv = self.total_value(prices)
            if tv == 0:
                return 0.0
            price = prices.get(token, 1.0) if prices else 1.0
            amt = self.balances.get(token, types.SimpleNamespace(amount=0)).amount
            return (amt * price) / tv

        def update_drawdown(self, prices):
            pass

        def record_prices(self, prices):
            pass

        def update_risk_metrics(self):
            pass

        def current_drawdown(self, prices):
            return 0.0

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

    return importlib.import_module("solhunter_zero.main")


def test_paper(monkeypatch):
    main_module = _load_main(monkeypatch)
    from solhunter_zero.simulation import SimulationResult  # noqa: E402
    mem = main_module.Memory("sqlite:///:memory:")
    pf = main_module.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ["TOK"]

    monkeypatch.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)

    async def fake_place_order(token, side, amount, price, **_):
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)
    import solhunter_zero.gas as gas_mod  # noqa: E402

    monkeypatch.setattr(gas_mod, "get_current_fee", lambda testnet=False: 0.0)

    async def _no_fee_async(*_a, **_k):
        return 0.0

    monkeypatch.setattr(gas_mod, "get_current_fee_async", _no_fee_async)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    trades = asyncio.run(mem.list_trades())
    assert len(trades) == 1
    assert trades[0].token == "TOK"
    assert pf.balances["TOK"].amount > 0


def test_paper_metrics(monkeypatch):
    main_module = _load_main(monkeypatch)
    from solhunter_zero.simulation import SimulationResult  # noqa: E402
    mem = main_module.Memory("sqlite:///:memory:")
    pf = main_module.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ["TOK"]

    monkeypatch.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
    )

    def patched_log_trade(self, **kw):
        if kw.get("direction") == "buy":
            kw["price"] = 1.0
        elif kw.get("direction") == "sell":
            kw["price"] = 2.0
        kw.setdefault("reason", "")
        self.trades.append(types.SimpleNamespace(**kw))

    monkeypatch.setattr(main_module.Memory, "log_trade", patched_log_trade)

    async def fake_place_order(token, side, amount, price, **_):
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)
    import solhunter_zero.gas as gas_mod  # noqa: E402

    monkeypatch.setattr(gas_mod, "get_current_fee", lambda testnet=False: 0.0)

    async def _no_fee_async(*_a, **_k):
        return 0.0

    monkeypatch.setattr(gas_mod, "get_current_fee_async", _no_fee_async)

    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: True)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    roi = TradeAnalyzer(mem).roi_by_agent()
    assert roi.get("", 0.0) == 1.0
