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
from .onchain_metrics import top_volume_tokens

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

    token_file: str | None = None,

    discovery_method: str = "websocket",

    keypair=None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    """Execute a single trading iteration asynchronously."""

    
    try:
        tokens = await scan_tokens_async(
            offline=offline, token_file=token_file, method=discovery_method
        )
    except TypeError:
        tokens = await scan_tokens_async(offline=offline, token_file=token_file)


    rpc_url = os.getenv("SOLANA_RPC_URL")
    if rpc_url and not offline:
        try:
            top = top_volume_tokens(rpc_url, limit=len(tokens))
            tokens = [t for t in tokens if t in top]
        except Exception as exc:  # pragma: no cover - network errors
            logging.warning("Volume ranking failed: %s", exc)


    # Run Monte Carlo simulations for discovered tokens and current holdings.
    tokens_to_sim = set(tokens) | set(portfolio.balances.keys())
    sim_tasks = {
        token: asyncio.to_thread(run_simulations, token, count=100)
        for token in tokens_to_sim
    }
    sim_results = {tok: await task for tok, task in sim_tasks.items()}

    # Fetch current prices for discovered tokens and any held positions.
    price_tokens = tokens_to_sim
    prices = await fetch_token_prices_async(price_tokens)

    for token, sims in sim_results.items():
        price = prices.get(token)
        if price is None:
            continue

        # Decision to buy new token
        if token in tokens and should_buy(sims):
            amount = 1.0
            await place_order_async(
                token,
                "buy",
                amount,
                price,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            memory.log_trade(
                token=token,
                direction="buy",
                amount=amount,
                price=price,
                reason="buy",
            )
            portfolio.add(token, amount, price)

        # Decision to sell based on simulations
        elif should_sell(sims) and token in portfolio.balances:
            amount = portfolio.balances[token].amount
            await place_order_async(
                token,
                "sell",
                amount,
                price,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            memory.log_trade(
                token=token,
                direction="sell",
                amount=amount,
                price=price,
                reason="sell",
            )
            portfolio.update(token, -amount, price)

    # Stop-loss and take-profit checks for held positions.
    for token, pos in list(portfolio.balances.items()):
        price = prices.get(token)
        if price is None:
            continue
        roi = portfolio.position_roi(token, price)

        if stop_loss is not None and roi <= -stop_loss:
            amount = pos.amount
            await place_order_async(
                token,
                "sell",
                amount,
                price,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            memory.log_trade(
                token=token,
                direction="sell",
                amount=amount,
                price=price,
                reason="stop-loss",
            )
            portfolio.update(token, -amount, price)

        elif take_profit is not None and roi >= take_profit:
            amount = pos.amount
            await place_order_async(
                token,
                "sell",
                amount,
                price,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )
            memory.log_trade(
                token=token,
                direction="sell",
                amount=amount,
                price=price,
                reason="take-profit",
            )
            portfolio.update(token, -amount, price)



def main(
    memory_path: str = "sqlite:///memory.db",
    loop_delay: int = 60,
    *,
    iterations: int | None = None,
    testnet: bool = False,
    dry_run: bool = False,
    offline: bool = False,

    discovery_method: str | None = None,

    token_file: str | None = None,

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

    token_file:
        Path to a file containing token addresses to scan.

    portfolio_path:
        Path to the JSON file for persisting portfolio state.




    """

    from .wallet import load_keypair

    cfg = apply_env_overrides(load_config(config_path))
    set_env_from_config(cfg)

    if discovery_method is None:
        discovery_method = cfg.get("discovery_method")
    if discovery_method is None:
        discovery_method = os.getenv("DISCOVERY_METHOD", "websocket")

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

                    token_file=token_file,
                    discovery_method=discovery_method,

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

                    token_file=token_file,
                    discovery_method=discovery_method,

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
        "--discovery-method",
        default=None,
        help="Token discovery method (websocket, onchain, pools, file)",
    )
    parser.add_argument(

        "--token-list",
        default=None,
        metavar="FILE",
        help="Read token addresses from FILE instead of querying the network",

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

        token_file=args.token_list,
        discovery_method=args.discovery_method,

        keypair_path=args.keypair,
        portfolio_path=args.portfolio_path,
        config_path=args.config,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )
