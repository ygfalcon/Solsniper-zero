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
    loaded = investor_demo.load_prices()
    if isinstance(loaded, dict):
        prices, _ = next(iter(loaded.values()))
    else:
        prices, _ = loaded
    expected = abs(prices[1] - prices[0]) if len(prices) >= 2 else 0.0
    profit = asyncio.run(investor_demo._demo_arbitrage())
    assert profit == pytest.approx(expected)
    assert investor_demo.used_trade_types == {"arbitrage"}


def test_demo_flash_loan():
    loaded = investor_demo.load_prices()
    if isinstance(loaded, dict):
        prices, _ = next(iter(loaded.values()))
    else:
        prices, _ = loaded
    principal = prices[0] if prices else 0.0
    expected = principal * investor_demo.FLASH_LOAN_INTEREST
    profit = asyncio.run(investor_demo._demo_flash_loan())
    assert profit == pytest.approx(expected)
    assert investor_demo.used_trade_types == {"flash_loan"}


def test_demo_sniper():
    loaded = investor_demo.load_prices(preset="multi")
    assert isinstance(loaded, dict)
    expected = [
        token
        for token, (prices, _dates) in loaded.items()
        if len(prices) >= 2 and prices[1] > prices[0]
    ]
    tokens = asyncio.run(investor_demo._demo_sniper())
    assert tokens == expected
    assert investor_demo.used_trade_types == {"sniper"}


def test_demo_dex_scanner():
    loaded = investor_demo.load_prices(preset="multi")
    assert isinstance(loaded, dict)
    expected = [f"{token}-USDC" for token in sorted(loaded.keys())]
    pools = asyncio.run(investor_demo._demo_dex_scanner())
    assert pools == expected
    assert investor_demo.used_trade_types == {"dex_scanner"}
