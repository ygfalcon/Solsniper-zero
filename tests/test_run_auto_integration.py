import sys
import types
import asyncio
import contextlib
import json
import os
import pytest
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
sys.modules['sklearn.cluster'] = types.SimpleNamespace(KMeans=object, DBSCAN=object)
_xgb_mod = types.ModuleType('xgboost')
_xgb_mod.__spec__ = importlib.machinery.ModuleSpec('xgboost', None)
_xgb_mod.XGBRegressor = object
sys.modules['xgboost'] = _xgb_mod

torch_mod = types.ModuleType('torch')
torch_mod.__spec__ = importlib.machinery.ModuleSpec('torch', None)
torch_mod.no_grad = contextlib.nullcontext
torch_mod.tensor = lambda *a, **k: None
torch_mod.nn = types.SimpleNamespace(
    LSTM=object,
    Linear=object,
    TransformerEncoder=object,
    TransformerEncoderLayer=object,
    Module=object,
)
torch_mod.optim = types.ModuleType('optim')
torch_mod.optim.__spec__ = importlib.machinery.ModuleSpec('torch.optim', None)
torch_mod.utils = types.ModuleType('utils')
torch_mod.utils.data = types.SimpleNamespace(Dataset=object, DataLoader=object)
sys.modules['torch.utils'] = torch_mod.utils
sys.modules['torch.utils.data'] = torch_mod.utils.data
sys.modules['torch.nn'] = torch_mod.nn
sys.modules['torch'] = torch_mod

_pl_mod = types.ModuleType('pytorch_lightning')
_pl_mod.callbacks = types.SimpleNamespace(Callback=object)
_pl_mod.LightningModule = object
_pl_mod.LightningDataModule = object
_pl_mod.Trainer = object
sys.modules.setdefault('pytorch_lightning', _pl_mod)


from solders.keypair import Keypair
import solhunter_zero.config as cfg_mod


def test_run_auto_integration(monkeypatch, tmp_path):
    with monkeypatch.context() as mp:
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
        from solhunter_zero import main as main_module
        from solhunter_zero.simulation import SimulationResult

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "cfg.toml"
        cfg_file.write_text("risk_tolerance = 0.5")
        (cfg_dir / "active").write_text("cfg.toml")
        monkeypatch.setattr(cfg_mod, "CONFIG_DIR", str(cfg_dir))
        monkeypatch.setattr(cfg_mod, "ACTIVE_CONFIG_FILE", str(cfg_dir / "active"))
        monkeypatch.setattr(main_module, "CONFIG_DIR", str(cfg_dir))

        keys_dir = tmp_path / "keys"
        keys_dir.mkdir()
        monkeypatch.setattr(main_module.wallet, "KEYPAIR_DIR", str(keys_dir))
        monkeypatch.setattr(
            main_module.wallet,
            "ACTIVE_KEYPAIR_FILE",
            str(keys_dir / "active"),
        )
        kp = Keypair()
        (keys_dir / "only.json").write_text(json.dumps(list(kp.to_bytes())))

        async def fake_discover(self, **kwargs):
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
            lambda token, count=100: [
                SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)
            ],
        )
        monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
        monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)

        async def fake_place_order(token, side, amount, price, **kwargs):
            return {"order_id": "1"}

        monkeypatch.setattr(main_module, "place_order_async", fake_place_order)

        async def fake_fetch(*_a, **_k):
            return {"liquidity": 0.0, "volume": 0.0}

        monkeypatch.setattr(
            main_module,
            "fetch_dex_metrics_async",
            fake_fetch,
        )
        monkeypatch.setattr(main_module, "ensure_connectivity", lambda **_: None)
        async def _noop(*_a, **_k):
            return None
        monkeypatch.setattr(main_module.event_bus, "start_ws_server", _noop)
        monkeypatch.setattr(main_module.event_bus, "stop_ws_server", _noop)

        mem_path = tmp_path / "mem.db"
        pf_path = tmp_path / "pf.json"

        main_module.run_auto(
            memory_path=f"sqlite:///{mem_path}",
            portfolio_path=str(pf_path),
            loop_delay=0,
            iterations=1,
            dry_run=False,
        )

        mem = main_module.Memory(f"sqlite:///{mem_path}")
        trades = asyncio.run(mem.list_trades())
        assert len(trades) == 1
        assert trades[0].token == "TOK"

        pf = main_module.Portfolio(path=str(pf_path))
        assert pf.balances.get("TOK") is not None
        assert pf.balances["TOK"].amount > 0
