import sys
import types
import importlib
import importlib.machinery


def _load_paper_test(monkeypatch):
    """Import ``scripts.paper_test`` with stubbed dependencies."""
    # Stub optional heavy dependencies used by solhunter modules
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

    memory_store: dict[str, list] = {}

    class DummyMemory:
        def __init__(self, url: str = "sqlite:///:memory:"):
            self.url = url
            memory_store.setdefault(url, [])

        def log_trade(self, **kw):
            memory_store[self.url].append(types.SimpleNamespace(**kw))

        def list_trades(self, limit: int | None = None):
            data = memory_store[self.url]
            return data[:limit] if limit is not None else data

        def log_var(self, value: float) -> None:
            pass

        def list_vars(self):
            return []

        def close(self):
            pass

    memory_mod = types.ModuleType("solhunter_zero.memory")
    memory_mod.Memory = DummyMemory
    monkeypatch.setitem(sys.modules, "solhunter_zero.memory", memory_mod)

    class DummyAnalyzer:
        def __init__(self, mem):
            self.mem = mem

        def roi_by_agent(self):
            return {}

    analyzer_mod = types.ModuleType("solhunter_zero.trade_analyzer")
    analyzer_mod.TradeAnalyzer = DummyAnalyzer
    monkeypatch.setitem(sys.modules, "solhunter_zero.trade_analyzer", analyzer_mod)

    util_mod = types.ModuleType("solhunter_zero.util")
    util_mod.run_coro = lambda x: x
    monkeypatch.setitem(sys.modules, "solhunter_zero.util", util_mod)

    http_mod = types.ModuleType("solhunter_zero.http")
    http_mod.close_session = lambda: None
    monkeypatch.setitem(sys.modules, "solhunter_zero.http", http_mod)

    main_mod = types.ModuleType("solhunter_zero.main")

    def stub_run_auto(*, memory_path: str, iterations: int, dry_run: bool, offline: bool, capital=None, **_):
        mem = DummyMemory(memory_path)
        if capital is None:
            # Overspend when capital is missing
            mem.log_trade(token="TOK", direction="buy", amount=200.0, price=1.0)
        else:
            half = float(capital) / 2.0
            mem.log_trade(token="TOK", direction="buy", amount=half, price=1.0)
            mem.log_trade(token="TOK", direction="buy", amount=half, price=1.0)

    main_mod.run_auto = stub_run_auto
    monkeypatch.setitem(sys.modules, "solhunter_zero.main", main_mod)

    paper_test = importlib.import_module("scripts.paper_test")
    return paper_test, memory_store


def test_paper_test_capital(monkeypatch):
    paper_test, store = _load_paper_test(monkeypatch)
    rc = paper_test.main([
        "--iterations",
        "5",
        "--memory",
        "sqlite:///:memory:",
        "--capital",
        "100",
    ])
    assert rc == 0
    trades = store.get("sqlite:///:memory:", [])
    spent = sum(float(t.amount) * float(t.price) for t in trades if t.direction == "buy")
    assert spent <= 100.0
