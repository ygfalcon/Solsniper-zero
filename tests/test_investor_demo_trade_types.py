import asyncio
import pytest

from solhunter_zero import investor_demo



@pytest.fixture(autouse=True)
def clear_used_trade_types():
    '''Ensure used_trade_types is reset for each test.'''
    investor_demo.used_trade_types.clear()
    yield
    investor_demo.used_trade_types.clear()


def test_demo_arbitrage():
    prices, _ = investor_demo.load_prices(preset="short")
    profit = asyncio.run(investor_demo._demo_arbitrage())
    expected = abs(prices[1] - prices[0]) if len(prices) > 1 else 0.0
    assert profit == pytest.approx(expected)
    assert investor_demo.used_trade_types == {"arbitrage"}


def test_demo_flash_loan():
    prices, _ = investor_demo.load_prices(preset="short")
    profit = asyncio.run(investor_demo._demo_flash_loan())
    expected = (
        abs(prices[2] - prices[1]) / prices[1] if len(prices) > 2 and prices[1] else 0.0
    )
    assert profit == pytest.approx(expected)
    assert investor_demo.used_trade_types == {"flash_loan"}


def test_demo_sniper():
    _, dates = investor_demo.load_prices(preset="short")
    tokens = asyncio.run(investor_demo._demo_sniper())
    expected = [f"token_{dates[0]}"] if dates else ["token_demo"]
    assert tokens == expected
    assert investor_demo.used_trade_types == {"sniper"}


def test_demo_dex_scanner():
    _, dates = investor_demo.load_prices(preset="short")
    pools = asyncio.run(investor_demo._demo_dex_scanner())
    expected = [f"pool_{dates[1]}"] if len(dates) > 1 else ["pool_demo"]
    assert pools == expected
    assert investor_demo.used_trade_types == {"dex_scanner"}
