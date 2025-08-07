import contextlib
import importlib.machinery
import sys
import types
from pathlib import Path

from tests.stubs import stub_numpy


def setup_live_trading_env(mp):
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
    mp.setitem(sys.modules, "faiss", faiss_mod)
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    st_mod.SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
        get_sentence_embedding_dimension=lambda: 1, encode=lambda x: []
    )
    mp.setitem(sys.modules, "sentence_transformers", st_mod)

    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
    mp.setitem(sys.modules, "sklearn", sklearn)
    mp.setitem(
        sys.modules,
        "sklearn.linear_model",
        types.SimpleNamespace(LinearRegression=object),
    )
    mp.setitem(
        sys.modules,
        "sklearn.ensemble",
        types.SimpleNamespace(
            GradientBoostingRegressor=object, RandomForestRegressor=object
        ),
    )
    mp.setitem(
        sys.modules,
        "sklearn.cluster",
        types.SimpleNamespace(KMeans=object, DBSCAN=object),
    )
    xgb_mod = types.ModuleType("xgboost")
    xgb_mod.__spec__ = importlib.machinery.ModuleSpec("xgboost", None)
    xgb_mod.XGBRegressor = object
    mp.setitem(sys.modules, "xgboost", xgb_mod)

    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch_mod.no_grad = contextlib.nullcontext
    torch_mod.tensor = lambda *a, **k: None
    torch_mod.nn = types.SimpleNamespace(
        LSTM=object,
        Linear=object,
        TransformerEncoder=object,
        TransformerEncoderLayer=object,
        Module=object,
    )
    torch_mod.optim = types.ModuleType("optim")
    torch_mod.optim.__spec__ = importlib.machinery.ModuleSpec("torch.optim", None)
    torch_mod.utils = types.ModuleType("utils")
    torch_mod.utils.data = types.SimpleNamespace(Dataset=object, DataLoader=object)
    mp.setitem(sys.modules, "torch.utils", torch_mod.utils)
    mp.setitem(sys.modules, "torch.utils.data", torch_mod.utils.data)
    mp.setitem(sys.modules, "torch.nn", torch_mod.nn)
    mp.setitem(sys.modules, "torch", torch_mod)

    pl_mod = types.ModuleType("pytorch_lightning")
    pl_mod.callbacks = types.SimpleNamespace(Callback=object)
    pl_mod.LightningModule = object
    pl_mod.LightningDataModule = object
    pl_mod.Trainer = object
    mp.setitem(sys.modules, "pytorch_lightning", pl_mod)

    trans_mod = types.ModuleType("transformers")
    trans_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    trans_mod.pipeline = lambda *a, **k: lambda *x, **y: None
    mp.setitem(sys.modules, "transformers", trans_mod)

    bip_mod = types.ModuleType("bip_utils")
    bip_mod.__spec__ = importlib.machinery.ModuleSpec("bip_utils", None)
    bip_mod.Bip39SeedGenerator = object
    bip_mod.Bip44 = object
    bip_mod.Bip44Coins = object
    bip_mod.Bip44Changes = object
    mp.setitem(sys.modules, "bip_utils", bip_mod)

    stub_models = types.ModuleType("solhunter_zero.models")
    stub_models.get_model = lambda *a, **k: None
    stub_models.load_compiled_model = lambda *a, **k: None
    stub_models.export_torchscript = lambda *a, **k: None
    mp.setitem(sys.modules, "solhunter_zero.models", stub_models)
    mp.setitem(
        sys.modules,
        "solhunter_zero.models.regime_classifier",
        types.SimpleNamespace(get_model=lambda *a, **k: None),
    )
    stub_models.regime_classifier = sys.modules[
        "solhunter_zero.models.regime_classifier"
    ]

    async def _dummy_async(*_a, **_k):
        return None

    event_bus_stub = types.SimpleNamespace(
        start_ws_server=_dummy_async,
        stop_ws_server=_dummy_async,
        subscribe=lambda *a, **k: (lambda: None),
        publish=lambda *a, **k: None,
    )

    class _Sub:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    event_bus_stub.subscription = lambda *a, **k: _Sub()
    event_bus_stub._reload_bus = lambda *a, **k: None
    event_bus_stub._reload_broker = lambda *a, **k: None
    event_bus_stub._reload_serialization = lambda *a, **k: None
    mp.setitem(sys.modules, "solhunter_zero.event_bus", event_bus_stub)
    wallet_stub = types.SimpleNamespace(
        load_keypair=lambda path: object(),
        load_selected_keypair=lambda: object(),
    )
    mp.setitem(sys.modules, "solhunter_zero.wallet", wallet_stub)
    metrics_stub = types.SimpleNamespace(
        start=lambda: None,
        publish=lambda *a, **k: None,
        emit_startup_complete=lambda *a, **k: None,
    )
    mp.setitem(sys.modules, "solhunter_zero.metrics_aggregator", metrics_stub)
    bootstrap_stub = types.SimpleNamespace(bootstrap=lambda: None)
    mp.setitem(sys.modules, "solhunter_zero.bootstrap", bootstrap_stub)
    token_scanner_stub = types.SimpleNamespace(scan_tokens_async=lambda *_a, **_k: [])
    mp.setitem(sys.modules, "solhunter_zero.token_scanner", token_scanner_stub)
    onchain_stub = types.SimpleNamespace(
        async_top_volume_tokens=lambda *_a, **_k: [],
        fetch_dex_metrics_async=lambda *_a, **_k: {"liquidity": 0.0, "volume": 0.0},
    )
    mp.setitem(sys.modules, "solhunter_zero.onchain_metrics", onchain_stub)
    exchange_stub = types.SimpleNamespace(place_order_async=lambda *a, **k: {})
    mp.setitem(sys.modules, "solhunter_zero.exchange", exchange_stub)
    order_book_ws_stub = types.SimpleNamespace()
    mp.setitem(sys.modules, "solhunter_zero.order_book_ws", order_book_ws_stub)
    agent_manager_stub = types.SimpleNamespace(
        AgentManager=type(
            "AgentManager", (), {"from_config": staticmethod(lambda cfg: None)}
        )
    )
    mp.setitem(sys.modules, "solhunter_zero.agent_manager", agent_manager_stub)

    class _DiscoveryAgent:
        async def discover_tokens(self, **_):
            return ["TOK"]

    discovery_stub = types.SimpleNamespace(DiscoveryAgent=_DiscoveryAgent)
    mp.setitem(sys.modules, "solhunter_zero.agents.discovery", discovery_stub)
    arbitrage_stub = types.SimpleNamespace(detect_and_execute_arbitrage=_dummy_async)
    mp.setitem(sys.modules, "solhunter_zero.arbitrage", arbitrage_stub)
    depth_client_stub = types.SimpleNamespace()
    mp.setitem(sys.modules, "solhunter_zero.depth_client", depth_client_stub)
    stub_numpy()
    aiohttp_mod = types.ModuleType("aiohttp")
    aiohttp_mod.ClientSession = object
    mp.setitem(sys.modules, "aiohttp", aiohttp_mod)
    aiofiles_mod = types.ModuleType("aiofiles")
    aiofiles_mod.open = contextlib.nullcontext
    mp.setitem(sys.modules, "aiofiles", aiofiles_mod)
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.ValidationError = Exception

    class _BaseModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def dict(self, *a, **k):
            return dict(self.__dict__)

    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.AnyUrl = str

    def _decorator(*_a, **_k):
        def _wrap(fn):
            return fn

        return _wrap

    pydantic_mod.root_validator = _decorator
    pydantic_mod.validator = _decorator
    mp.setitem(sys.modules, "pydantic", pydantic_mod)
    from solhunter_zero import main as main_module
    from solhunter_zero.simulation import SimulationResult

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "config" / "default.toml"
    key_path = repo_root / "keypairs" / "default.json"

    # Load preset config and default keypair via environment
    mp.setenv("SOLHUNTER_CONFIG", str(cfg_path))
    mp.setenv("KEYPAIR_PATH", str(key_path))
    mp.setenv("AGENTS", "")
    mp.setenv("USE_DEPTH_STREAM", "0")

    # Avoid network and slow components
    async def fake_discover(self, **_):
        return ["TOK"]

    mp.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "buy", "amount": 1, "price": 0}]

        def list_missing(self):
            return []

    mp.setattr(main_module, "StrategyManager", DummySM)
    mp.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)
        ],
    )
    mp.setattr(main_module, "should_buy", lambda sims: True)
    mp.setattr(main_module, "should_sell", lambda sims, **k: False)

    async def fake_fetch(*_a, **_k):
        return {"liquidity": 0.0, "volume": 0.0}

    mp.setattr(
        main_module,
        "fetch_dex_metrics_async",
        fake_fetch,
    )
    mp.setattr(main_module, "_start_depth_service", lambda cfg: None)

    async def _noop_ws(*_a, **_k):
        return None

    mp.setattr(main_module.event_bus, "start_ws_server", _noop_ws)

    # Capture Memory usage
    trades: list[dict] = []
    mem_inst: dict = {}
    orig_init = main_module.Memory.__init__

    def fake_init(self, url: str = "sqlite:///:memory:"):
        orig_init(self, url=url)
        mem_inst["inst"] = self

    mp.setattr(main_module.Memory, "__init__", fake_init)

    def fake_log_trade(self, **kw):
        trades.append(kw)

    mp.setattr(main_module.Memory, "log_trade", fake_log_trade)
    mp.setattr(main_module.Memory, "start_writer", lambda *a, **k: None)
    mp.setattr(main_module.Memory, "wait_ready", _dummy_async)
    mp.setattr(main_module.Portfolio, "update", lambda *a, **k: None)
    mp.setattr(main_module.asyncio, "sleep", lambda *_a, **_k: None)

    async def fake_place_order(token, side, amount, price, **kwargs):
        if mem_inst.get("inst"):
            mem_inst["inst"].log_trade(
                token=token, direction=side, amount=amount, price=price
            )
        return {"order_id": "1", "dry_run": kwargs.get("dry_run")}

    mp.setattr(main_module, "place_order_async", fake_place_order)

    return main_module, trades


def test_live_trading(monkeypatch):
    with monkeypatch.context() as mp:
        main_module, trades = setup_live_trading_env(mp)

        # Run a single iteration in dry-run mode
        main_module.main(
            iterations=1,
            dry_run=True,
            loop_delay=0,
            memory_path="sqlite:///:memory:",
        )

        # Ensure a trade attempt was recorded
        assert trades
