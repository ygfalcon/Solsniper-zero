import asyncio
import importlib
import pkgutil
from typing import List

import pytest

from solhunter_zero.portfolio import Portfolio
from solhunter_zero.strategy_manager import StrategyManager


def _discover_strategy_modules() -> List[object]:
    """Import all ``solhunter_zero`` modules defining a top-level ``evaluate``."""

    import pathlib
    import solhunter_zero

    modules: List[object] = []
    base = pathlib.Path(solhunter_zero.__file__).parent
    prefix = solhunter_zero.__name__ + "."

    for info in pkgutil.iter_modules([str(base)]):
        module_name = prefix + info.name
        file_path = base / f"{info.name}.py"
        try:
            src = file_path.read_text()
        except Exception:  # pragma: no cover - invalid file
            continue
        if "async def evaluate(token" not in src:
            continue
        try:
            mod = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional deps
            continue
        modules.append(mod)

    return modules


def _prepare_module(monkeypatch: pytest.MonkeyPatch, mod: object) -> None:
    """Monkeypatch helpers so ``mod.evaluate`` runs without external deps."""

    name = getattr(mod, "__name__", "")

    if name.endswith("sniper"):
        class _DummyRes:
            expected_roi = 1.0
            volatility = 0.0
            volume_spike = 1.0
            depth_change = 0.0
            whale_activity = 0.0
            tx_rate = 1.0

        monkeypatch.setattr(
            mod,
            "run_simulations",
            lambda token, count=100: [_DummyRes()],
        )
        monkeypatch.setattr(mod, "should_buy", lambda sims: True)
        monkeypatch.setattr(mod, "should_sell", lambda sims, **kw: False)

        async def _prices(tokens):
            return {t: 1.0 for t in tokens}

        monkeypatch.setattr(mod, "fetch_token_prices_async", _prices)
        monkeypatch.setattr(mod, "predict_price_movement", lambda token: 0.1)
        monkeypatch.setattr(mod, "dynamic_order_size", lambda *a, **k: 1.0)

    elif name.endswith("arbitrage"):
        async def _arb(token, **kwargs):
            return (0, 1)

        monkeypatch.setattr(mod, "detect_and_execute_arbitrage", _arb)
        monkeypatch.setenv("ARBITRAGE_THRESHOLD", "0.01")
        monkeypatch.setenv("ARBITRAGE_AMOUNT", "2")

    else:  # pragma: no cover - future strategies
        async def _dummy(token, portfolio):
            return [{"token": token, "side": name, "amount": 1.0, "price": 0.0}]

        monkeypatch.setattr(mod, "evaluate", _dummy)


def test_one_click_all_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = _discover_strategy_modules()
    for mod in modules:
        _prepare_module(monkeypatch, mod)

    portfolio = Portfolio(path=None)
    mgr = StrategyManager([m.__name__ for m in modules])

    actions = asyncio.run(mgr.evaluate("TKN", portfolio))

    assert len(actions) == len(modules)

