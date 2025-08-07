import asyncio
import pytest

from solhunter_zero import investor_demo



@pytest.fixture(autouse=True)
def clear_used_trade_types():
    '''Ensure used_trade_types is reset for each test.'''
    investor_demo.used_trade_types.clear()
    yield
    investor_demo.used_trade_types.clear()


def test_demo_arbitrage(monkeypatch):
    from solhunter_zero import arbitrage

    def fake_best_route(*a, **k):
        return ["dex1", "dex2"], 4.795

    monkeypatch.setattr(arbitrage, "_best_route", fake_best_route)
    res = asyncio.run(investor_demo._demo_arbitrage())
    assert res["path"] == ["dex1", "dex2"]
    assert res["profit"] == pytest.approx(4.795)
    assert investor_demo.used_trade_types == {"arbitrage"}


def test_demo_arbitrage_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("arbitrage"):
            raise ImportError("no module")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = asyncio.run(investor_demo._demo_arbitrage())
    assert res == {"path": [], "profit": 0.0}
    assert investor_demo.used_trade_types == {"arbitrage"}


def test_demo_flash_loan():
    sig = asyncio.run(investor_demo._demo_flash_loan())
    assert sig == "demo_sig"
    assert investor_demo.used_trade_types == {"flash_loan"}


def test_demo_sniper():
    tokens = asyncio.run(investor_demo._demo_sniper())
    assert investor_demo.used_trade_types == {"sniper"}
    assert tokens in ([], ["TKN"])


def test_demo_dex_scanner():
    pools = asyncio.run(investor_demo._demo_dex_scanner())
    assert pools == ["mintA", "mintB"]
    assert investor_demo.used_trade_types == {"dex_scanner"}


def test_demo_route_ffi(monkeypatch):
    from solhunter_zero import routeffi as rffi

    monkeypatch.setattr(
        rffi, "_best_route_json", lambda *a, **k: (["x", "y"], 2.0)
    )
    res = asyncio.run(investor_demo._demo_route_ffi())
    assert res["path"] == ["x", "y"]
    assert res["profit"] == pytest.approx(2.0)
    assert investor_demo.used_trade_types == {"route_ffi"}
