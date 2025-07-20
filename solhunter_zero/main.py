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

from .simulation import run_simulations
from .decision import should_buy, should_sell
from .memory import Memory
from .portfolio import Portfolio
from .exchange import place_order_async

logging.basicConfig(level=logging.INFO)


async def _run_iteration(
    memory: Memory,
    portfolio: Portfolio,
    *,
    testnet: bool = False,
    dry_run: bool = False,
    offline: bool = False,
    keypair=None,
) -> None:
    """Execute a single trading iteration asynchronously."""
    tokens = await scan_tokens_async(offline=offline)

    for token in tokens:
        sims = run_simulations(token, count=100)
        if should_buy(sims):
            logging.info("Buying %s", token)
            await place_order_async(
                token,
                side="buy",
                amount=1,
                price=0,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            if not dry_run:
                memory.log_trade(token=token, direction="buy", amount=1, price=0)
                portfolio.update(token, 1, 0)

    for token, pos in list(portfolio.balances.items()):
        sims = run_simulations(token, count=100)
        if should_sell(sims):
            logging.info("Selling %s", token)
            await place_order_async(
                token,
                side="sell",
                amount=pos.amount,
                price=0,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            if not dry_run:
                memory.log_trade(token=token, direction="sell", amount=pos.amount, price=0)
                portfolio.update(token, -pos.amount, 0)


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
    )
