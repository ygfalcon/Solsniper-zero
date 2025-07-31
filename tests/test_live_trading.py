from pathlib import Path
import sys
import types
import contextlib
import importlib.machinery

_faiss_mod = types.ModuleType('faiss')
_faiss_mod.__spec__ = importlib.machinery.ModuleSpec('faiss', None)
sys.modules.setdefault('faiss', _faiss_mod)
_st_mod = types.ModuleType('sentence_transformers')
_st_mod.__spec__ = importlib.machinery.ModuleSpec('sentence_transformers', None)
sys.modules.setdefault('sentence_transformers', _st_mod)
sys.modules['sentence_transformers'].SentenceTransformer = lambda *a, **k: types.SimpleNamespace(get_sentence_embedding_dimension=lambda:1, encode=lambda x: [])

sklearn = types.ModuleType('sklearn')
sklearn.__spec__ = importlib.machinery.ModuleSpec('sklearn', None)
sys.modules.setdefault('sklearn', sklearn)
sys.modules['sklearn.linear_model'] = types.SimpleNamespace(LinearRegression=object)
sys.modules['sklearn.ensemble'] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
_xgb_mod = types.ModuleType('xgboost')
_xgb_mod.__spec__ = importlib.machinery.ModuleSpec('xgboost', None)
_xgb_mod.XGBRegressor = object
sys.modules['xgboost'] = _xgb_mod

torch_mod = types.ModuleType('torch')
torch_mod.__spec__ = importlib.machinery.ModuleSpec('torch', None)
torch_mod.no_grad = contextlib.nullcontext
torch_mod.tensor = lambda *a, **k: None
torch_mod.nn = types.SimpleNamespace(LSTM=object, Linear=object, TransformerEncoder=object, TransformerEncoderLayer=object)
torch_mod.optim = types.ModuleType('optim')
torch_mod.optim.__spec__ = importlib.machinery.ModuleSpec('torch.optim', None)
sys.modules['torch'] = torch_mod

sys.modules['solhunter_zero.models'] = types.SimpleNamespace(get_model=lambda *a, **k: None)
_trans_mod = types.ModuleType('transformers')
_trans_mod.__spec__ = importlib.machinery.ModuleSpec('transformers', None)
_trans_mod.pipeline = lambda *a, **k: lambda *x, **y: None
sys.modules.setdefault('transformers', _trans_mod)
_bip_mod = types.ModuleType('bip_utils')
_bip_mod.__spec__ = importlib.machinery.ModuleSpec('bip_utils', None)
sys.modules.setdefault('bip_utils', _bip_mod)
sys.modules['bip_utils'].Bip39SeedGenerator = object
sys.modules['bip_utils'].Bip44 = object
sys.modules['bip_utils'].Bip44Coins = object
sys.modules['bip_utils'].Bip44Changes = object

from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult


def test_live_trading(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "config.highrisk.toml"
    key_path = repo_root / "keypairs" / "default.json"

    # Load preset config and default keypair via environment
    monkeypatch.setenv("SOLHUNTER_CONFIG", str(cfg_path))
    monkeypatch.setenv("KEYPAIR_PATH", str(key_path))
    monkeypatch.setenv("AGENTS", "")

    # Avoid network and slow components
    async def fake_discover(self, **_):
        return ["TOK"]

    monkeypatch.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "buy", "amount": 1, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)
    async def fake_fetch(*_a, **_k):
        return {"liquidity": 0.0, "volume": 0.0}

    monkeypatch.setattr(
        main_module,
        "fetch_dex_metrics_async",
        fake_fetch,
    )
    monkeypatch.setattr(main_module, "_start_depth_service", lambda cfg: None)

    # Capture Memory usage
    trades: list[dict] = []
    mem_inst: dict = {}
    orig_init = main_module.Memory.__init__

    def fake_init(self, url: str = "sqlite:///:memory:"):
        orig_init(self, url=url)
        mem_inst["inst"] = self

    monkeypatch.setattr(main_module.Memory, "__init__", fake_init)

    def fake_log_trade(self, **kw):
        trades.append(kw)

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)
    monkeypatch.setattr(main_module.asyncio, "sleep", lambda *_a, **_k: None)

    async def fake_place_order(token, side, amount, price, **kwargs):
        if mem_inst.get("inst"):
            mem_inst["inst"].log_trade(token=token, direction=side, amount=amount, price=price)
        return {"order_id": "1", "dry_run": kwargs.get("dry_run")}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)

    # Run a single iteration in dry-run mode
    main_module.main(iterations=1, dry_run=True, loop_delay=0, memory_path="sqlite:///:memory:")

    # Ensure a trade attempt was recorded
    assert trades
