import asyncio
import sys
import types
import numpy as np
import importlib.machinery
import pytest
from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult


pytest.importorskip("torch.nn.utils.rnn")
pytest.importorskip("transformers")

_faiss_mod = types.ModuleType("faiss")
_faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
sys.modules.setdefault("faiss", _faiss_mod)
_st_mod = types.ModuleType("sentence_transformers")
_st_mod.__spec__ = importlib.machinery.ModuleSpec(
    "sentence_transformers", None
)
sys.modules.setdefault("sentence_transformers", _st_mod)
sys.modules["sentence_transformers"].SentenceTransformer = (
    lambda *a, **k: types.SimpleNamespace(
        get_sentence_embedding_dimension=lambda: 1,
        encode=lambda x: np.zeros((1, 1)),
    )
)
sklearn = types.ModuleType("sklearn")
sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
sys.modules.setdefault("sklearn", sklearn)
sys.modules["sklearn.linear_model"] = types.SimpleNamespace(
    LinearRegression=object
)
sys.modules["sklearn.ensemble"] = types.SimpleNamespace(
    GradientBoostingRegressor=object, RandomForestRegressor=object
)
_xgb_mod = types.ModuleType("xgboost")
_xgb_mod.__spec__ = importlib.machinery.ModuleSpec("xgboost", None)
_xgb_mod.XGBRegressor = object
sys.modules["xgboost"] = _xgb_mod


def test_full_workflow(monkeypatch):
    mem = main_module.Memory("sqlite:///:memory:")
    pf = main_module.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ["TOK"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover
    )

    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)
        ],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)

    async def fake_place_order(token, side, amount, price, **_):
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)
    import solhunter_zero.gas as gas_mod

    monkeypatch.setattr(gas_mod, "get_current_fee", lambda testnet=False: 0.0)

    async def _no_fee_async(*_a, **_k):
        return 0.0

    monkeypatch.setattr(gas_mod, "get_current_fee_async", _no_fee_async)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False))

    trades = asyncio.run(mem.list_trades())
    assert len(trades) == 1
    assert trades[0].token == "TOK"
    assert pf.balances["TOK"].amount > 0
