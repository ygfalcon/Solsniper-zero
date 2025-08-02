import asyncio
import os
import json
from importlib import resources


async def run_paper_test(iterations: int = 100):
    """Run paper trading simulation using offline tick data."""
    os.environ.setdefault("USE_FLASH_LOANS", "1")

    from solhunter_zero.trade_analyzer import TradeAnalyzer
    from solhunter_zero.simulation import SimulationResult
    import solhunter_zero.main as main
    import solhunter_zero.gas as gas
    import solhunter_zero.arbitrage as arb

    # Load dataset
    data_path = resources.files("solhunter_zero") / "data" / "paper_test_ticks.json"
    with resources.as_file(data_path) as p:
        with p.open("r", encoding="utf-8") as fh:
            ticks = json.load(fh)

    mem = main.Memory("sqlite:///:memory:")
    pf = main.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ["TOK"]

    main.DiscoveryAgent.discover_tokens = fake_discover
    main.run_simulations = lambda token, count=100: [
        SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)
    ]
    main.should_buy = lambda sims: True
    main.should_sell = lambda sims, **k: True

    async def fake_place_order(token, side, amount, price, **_):
        return {"order_id": "1"}

    main.place_order_async = fake_place_order

    gas.get_current_fee = lambda testnet=False: 0.0

    async def _no_fee_async(*_a, **_k):
        return 0.0

    gas.get_current_fee_async = _no_fee_async

    async def fake_arb(*_a, **_k):
        return None

    arb.detect_and_execute_arbitrage = fake_arb

    for i in range(iterations):
        tick = ticks[i % len(ticks)]
        await main._run_iteration(
            mem,
            pf,
            dry_run=True,
            offline=True,
            arbitrage_threshold=0.01,
            arbitrage_amount=1.0,
        )
        direction = "buy" if i % 2 == 0 else "sell"
        await mem.log_trade(
            token="TOK",
            direction=direction,
            amount=1.0,
            price=tick["price"],
            reason="",
        )

    roi = TradeAnalyzer(mem).roi_by_agent()
    print(roi)
    return mem, roi


if __name__ == "__main__":
    asyncio.run(run_paper_test())
