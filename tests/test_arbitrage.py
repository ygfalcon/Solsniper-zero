import asyncio
from solhunter_zero import arbitrage as arb
from solhunter_zero.arbitrage import detect_and_execute_arbitrage


def test_arbitrage_executes_orders(monkeypatch):
    async def feed1(token):
        return 1.0

    async def feed2(token):
        return 1.2

    called = []

    async def fake_place_order(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called.append((side, price))
        return {"ok": True}

    monkeypatch.setattr('solhunter_zero.arbitrage.place_order_async', fake_place_order)

    result = asyncio.run(
        detect_and_execute_arbitrage('tok', [feed1, feed2], threshold=0.1, amount=5)
    )

    assert result == (0, 1)
    assert ('buy', 1.0) in called
    assert ('sell', 1.2) in called


def test_no_arbitrage(monkeypatch):
    async def feed1(token):
        return 1.0

    async def feed2(token):
        return 1.05

    called = {}

    async def fake_place_order(*a, **k):
        called['called'] = True
        return {}

    monkeypatch.setattr('solhunter_zero.arbitrage.place_order_async', fake_place_order)

    result = asyncio.run(
        detect_and_execute_arbitrage('tok', [feed1, feed2], threshold=0.1, amount=5)
    )

    assert result is None
    assert 'called' not in called


def test_default_price_feeds(monkeypatch):
    """Uses built-in Orca and Raydium feeds when none are provided."""

    async def fake_orca(token):
        return 1.5

    async def fake_raydium(token):
        return 1.8

    orders = []

    async def fake_place_order(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        orders.append((side, price))
        return {"ok": True}

    monkeypatch.setattr(arb, "fetch_orca_price_async", fake_orca)
    monkeypatch.setattr(arb, "fetch_raydium_price_async", fake_raydium)
    monkeypatch.setattr(arb, "place_order_async", fake_place_order)

    result = asyncio.run(detect_and_execute_arbitrage("tok", None, threshold=0.1, amount=2))

    assert result == (0, 1)
    assert ("buy", 1.5) in orders
    assert ("sell", 1.8) in orders
