import asyncio
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

import tests.stubs  # noqa: F401  # ensures heavy deps are stubbed

_faiss_mod = types.ModuleType("faiss")
_faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
sys.modules.setdefault("faiss", _faiss_mod)

_st_mod = types.ModuleType("sentence_transformers")
_st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
sys.modules.setdefault("sentence_transformers", _st_mod)
sys.modules[
    "sentence_transformers"
].SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
    get_sentence_embedding_dimension=lambda: 1,
    encode=lambda x: np.zeros((1, 1)),
)

sklearn = types.ModuleType("sklearn")
sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", None)
sys.modules.setdefault("sklearn", sklearn)
sys.modules["sklearn.linear_model"] = types.SimpleNamespace(LinearRegression=object)
sys.modules["sklearn.ensemble"] = types.SimpleNamespace(
    GradientBoostingRegressor=object, RandomForestRegressor=object
)

_xgb_mod = types.ModuleType("xgboost")
_xgb_mod.__spec__ = importlib.machinery.ModuleSpec("xgboost", None)
_xgb_mod.XGBRegressor = object
sys.modules.setdefault("xgboost", _xgb_mod)

if importlib.util.find_spec("solana.rpc.websocket_api") is None:
    ws_mod = types.ModuleType("solana.rpc.websocket_api")
    ws_mod.__spec__ = importlib.machinery.ModuleSpec("solana.rpc.websocket_api", None)
    ws_mod.connect = lambda *a, **k: None
    ws_mod.RpcTransactionLogsFilterMentions = object
    sys.modules.setdefault("solana.rpc.websocket_api", ws_mod)
else:
    import solana.rpc.websocket_api as ws_mod

    ws_mod.connect = lambda *a, **k: None
    if not hasattr(ws_mod, "RpcTransactionLogsFilterMentions"):
        ws_mod.RpcTransactionLogsFilterMentions = object

from solhunter_zero import arbitrage, flash_loans
from solhunter_zero import main as main_module
from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.simulation import SimulationResult
from solhunter_zero.trade_analyzer import TradeAnalyzer


def test_paper_trading_extended(monkeypatch):
    ticks = load_sample_ticks(Path("solhunter_zero/data/paper_ticks.json"))
    mem = main_module.Memory("sqlite:///:memory:")
    pf = main_module.Portfolio(path=None)

    orig_list_trades = mem.list_trades

    def list_trades_sync(*a, **k):
        return asyncio.run(orig_list_trades(*a, **k))

    monkeypatch.setattr(mem, "list_trades", list_trades_sync)

    tick_iter = iter(ticks)

    async def fake_discover(self, **kwargs):
        return ["TOK"]

    monkeypatch.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)

    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)
        ],
    )

    buy_next = [True]

    def should_buy_stub(sims):
        return buy_next[0] and "TOK" not in pf.balances

    def should_sell_stub(sims, **_):
        return (not buy_next[0]) and "TOK" in pf.balances

    monkeypatch.setattr(main_module, "should_buy", should_buy_stub)
    monkeypatch.setattr(main_module, "should_sell", should_sell_stub)

    async def fake_place_order(token, side, amount, price, **_):
        tick = next(tick_iter)
        trade_price = tick["price"]
        await mem.log_trade(
            token=token,
            direction=side,
            amount=amount,
            price=trade_price,
            reason="test",
        )
        amt = amount if side == "buy" else -amount
        await pf.update_async(token, amt, trade_price)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)

    calls = {"arbitrage": 0, "borrow": 0, "repay": 0}

    async def fake_borrow(amount, token, *a, **k):
        calls["borrow"] += 1
        return "sig_b"

    async def fake_repay(amount, token, *a, **k):
        calls["repay"] += 1
        return "sig_r"

    monkeypatch.setattr(flash_loans, "borrow_flash", fake_borrow)
    monkeypatch.setattr(flash_loans, "repay_flash", fake_repay)

    async def fake_arbitrage(token, threshold=0.0, amount=0.0, **k):
        calls["arbitrage"] += 1
        await flash_loans.borrow_flash(amount, token)
        await flash_loans.repay_flash(amount, token)

    monkeypatch.setattr(arbitrage, "detect_and_execute_arbitrage", fake_arbitrage)

    for _ in range(100):
        asyncio.run(
            main_module._run_iteration(
                mem,
                pf,
                dry_run=True,
                offline=True,
                arbitrage_threshold=0.1,
                arbitrage_amount=1.0,
            )
        )
        buy_next[0] = not buy_next[0]

    roi = TradeAnalyzer(mem).roi_by_agent().get("test", 0.0)
    assert roi != 0.0, f"ROI {roi} should be non-zero"

    assert calls["arbitrage"] > 0
    assert calls["borrow"] > 0
    assert calls["repay"] > 0
