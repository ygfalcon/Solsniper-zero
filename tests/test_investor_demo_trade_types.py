import asyncio
from unittest.mock import patch

import pytest

from solhunter_zero import investor_demo



@pytest.fixture(autouse=True)
def clear_used_trade_types():
    '''Ensure used_trade_types is reset for each test.'''
    investor_demo.used_trade_types.clear()
    yield
    investor_demo.used_trade_types.clear()


def test_demo_arbitrage():
    profit = asyncio.run(investor_demo._demo_arbitrage())
    assert profit == pytest.approx(0.51)
    assert investor_demo.used_trade_types == {"arbitrage"}


def test_demo_flash_loan():
    from solhunter_zero import flash_loans

    async def run() -> float:
        with patch.object(
            flash_loans, "borrow_flash", wraps=flash_loans.borrow_flash
        ) as mock_borrow, patch.object(
            flash_loans, "repay_flash", wraps=flash_loans.repay_flash
        ) as mock_repay:
            profit = await investor_demo._demo_flash_loan()
            assert mock_borrow.await_count == 1
            assert mock_repay.await_count == 1
            return profit

    profit = asyncio.run(run())
    assert profit == pytest.approx(0.001482213438735234)
    assert investor_demo.used_trade_types == {"flash_loan"}


def test_demo_sniper():
    tokens = asyncio.run(investor_demo._demo_sniper())
    assert tokens == ["token_2023-01-01"]
    assert investor_demo.used_trade_types == {"sniper"}


def test_demo_dex_scanner():
    pools = asyncio.run(investor_demo._demo_dex_scanner())
    assert pools == ["pool_2023-01-02"]
    assert investor_demo.used_trade_types == {"dex_scanner"}
