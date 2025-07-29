import sys
import types
import contextlib
import json
import os
import pytest

sys.modules.setdefault('faiss', types.ModuleType('faiss'))
sys.modules.setdefault('sentence_transformers', types.ModuleType('sentence_transformers'))
sys.modules['sentence_transformers'].SentenceTransformer = lambda *a, **k: types.SimpleNamespace(get_sentence_embedding_dimension=lambda:1, encode=lambda x: [])

sklearn = types.ModuleType('sklearn')
sys.modules.setdefault('sklearn', sklearn)
sys.modules['sklearn.linear_model'] = types.SimpleNamespace(LinearRegression=object)
sys.modules['sklearn.ensemble'] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
sys.modules['xgboost'] = types.ModuleType('xgboost')
sys.modules['xgboost'].XGBRegressor = object

torch_mod = types.ModuleType('torch')
torch_mod.no_grad = contextlib.nullcontext
torch_mod.tensor = lambda *a, **k: None
torch_mod.nn = types.SimpleNamespace(LSTM=object, Linear=object, TransformerEncoder=object, TransformerEncoderLayer=object)
torch_mod.optim = types.ModuleType('optim')
sys.modules['torch'] = torch_mod

sys.modules['solhunter_zero.models'] = types.SimpleNamespace(get_model=lambda *a, **k: None)

from solders.keypair import Keypair
from solhunter_zero import main as main_module
import solhunter_zero.config as cfg_mod
from solhunter_zero.simulation import SimulationResult


def test_run_auto_integration(monkeypatch, tmp_path):
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
    monkeypatch.setattr(main_module.wallet, "ACTIVE_KEYPAIR_FILE", str(keys_dir / "active"))
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
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
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
    trades = mem.list_trades()
    assert len(trades) == 1
    assert trades[0].token == "TOK"

    pf = main_module.Portfolio(path=str(pf_path))
    assert pf.balances.get("TOK") is not None
    assert pf.balances["TOK"].amount > 0
