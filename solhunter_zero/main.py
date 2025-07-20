import logging
import os
import asyncio
from argparse import ArgumentParser

from .config import load_config, apply_env_overrides, set_env_from_config

# Load configuration at startup so modules relying on environment variables
# pick up the values from config files or environment.
_cfg = apply_env_overrides(load_config())
set_env_from_config(_cfg)


from .scanner import scan_tokens_async
from .prices import fetch_token_prices_async

from .simulation import run_simulations
from .decision import should_buy, should_sell
from .memory import Memory
from .portfolio import Portfolio
from .exchange import place_order_async
from .prices import fetch_token_prices_async

logging.basicConfig(level=logging.INFO)


async def _run_iteration(
    memory: Memory,
    portfolio: Portfolio,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    offline: bool = False,
    keypair=None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    """Execute a single trading iteration asynchronously."""
    tokens = await scan_tokens_async(offline=offline)


    # run simulations for scanned tokens in parallel
    buy_tasks = {token: asyncio.to_thread(run_simulations, token, count=100) for token in tokens}
    buy_results = await asyncio.gather(*buy_tasks.values())
    tokens_to_buy = [tok for tok, sims in zip(buy_tasks.keys(), buy_results) if should_buy(sims)]

    # run simulations for currently held tokens in parallel
    positions = list(portfolio.balances.items())
    sell_tasks = {token: asyncio.to_thread(run_simulations, token, count=100) for token, _ in positions}
    sell_results = await asyncio.gather(*sell_tasks.values())
    tokens_to_sell = [
        token
        for (token, pos), sims in zip(positions, sell_results)
        if should_sell(sims)
    ]

    all_tokens = tokens_to_buy + tokens_to_sell
    prices = await fetch_token_prices_async(all_tokens) if all_tokens else {}

    for token in tokens_to_buy:
        price = prices.get(token, 0)
        logging.info("Buying %s", token)
        await place_order_async(
            token,
            side="buy",
            amount=1,
            price=price,
            testnet=testnet,
            dry_run=dry_run,
            keypair=keypair,
        )
        if not dry_run:
            memory.log_trade(token=token, direction="buy", amount=1, price=price)
            portfolio.update(token, 1, price)

    for token in tokens_to_sell:
        pos = portfolio.balances[token]
        price = prices.get(token, 0)
        logging.info("Selling %s", token)
        await place_order_async(
            token,
            side="sell",
            amount=pos.amount,
            price=price,
            testnet=testnet,
            dry_run=dry_run,
            keypair=keypair,
        )
        if not dry_run:
            memory.log_trade(token=token, direction="sell", amount=pos.amount, price=price)
            portfolio.update(token, -pos.amount, price)
=======
    for token in tokens:
        sims = run_simulations(token, count=100)



def main(
    memory_path: str = "sqlite:///memory.db",
    loop_delay: int = 60,
    *,
    iterations: int | None = None,
    testnet: bool = False,
    dry_run: bool = False,
    offline: bool = False,
    keypair_path: str | None = None,
    portfolio_path: str = "portfolio.json",
    config_path: str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    """Run the trading loop.

    Parameters
    ----------
    memory_path:
        Database URL for storing trades.
    loop_delay:
        Delay between iterations in seconds.




    iterations:
        Number of iterations to run before exiting. ``None`` runs forever.
    offline:
        Return a predefined token list instead of querying the network.
    portfolio_path:
        Path to the JSON file for persisting portfolio state.




    """

    from .wallet import load_keypair

    cfg = apply_env_overrides(load_config(config_path))
    set_env_from_config(cfg)

    if stop_loss is None:
        stop_loss = cfg.get("stop_loss")
    if take_profit is None:
        take_profit = cfg.get("take_profit")

    memory = Memory(memory_path)
    portfolio = Portfolio(path=portfolio_path)

    keypair = load_keypair(keypair_path) if keypair_path else None

    async def loop() -> None:
        if iterations is None:
            while True:
                await _run_iteration(
                    memory,
                    portfolio,
                    testnet=testnet,
                    dry_run=dry_run,
                    offline=offline,
                    keypair=keypair,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                await asyncio.sleep(loop_delay)
        else:
            for i in range(iterations):
                await _run_iteration(
                    memory,
                    portfolio,
                    testnet=testnet,
                    dry_run=dry_run,
                    offline=offline,
                    keypair=keypair,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                if i < iterations - 1:
                    await asyncio.sleep(loop_delay)

    asyncio.run(loop())





if __name__ == "__main__":
    parser = ArgumentParser(description="Run SolHunter Zero bot")
    parser.add_argument(
        "--memory-path",
        default="sqlite:///memory.db",
        help="Database URL for storing trades",
    )
    parser.add_argument(
        "--loop-delay",
        type=int,
        default=60,
        help="Delay between iterations in seconds",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run before exiting",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet DEX endpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders, just simulate",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use a static token list and skip network requests",
    )
    parser.add_argument(
        "--keypair",
        default=os.getenv("KEYPAIR_PATH"),
        help="Path to a JSON keypair for signing transactions",
    )
    parser.add_argument(
        "--portfolio-path",
        default="portfolio.json",
        help="Path to a JSON file for persisting portfolio state",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a configuration file",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss threshold as a fraction (e.g. 0.1 for 10%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit threshold as a fraction",
    )
    args = parser.parse_args()
    main(
        memory_path=args.memory_path,
        loop_delay=args.loop_delay,
        iterations=args.iterations,
        testnet=args.testnet,
        dry_run=args.dry_run,
        offline=args.offline,
        keypair_path=args.keypair,
        portfolio_path=args.portfolio_path,
        config_path=args.config,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )
