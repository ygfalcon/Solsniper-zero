import logging
import os
import asyncio
import contextlib
from argparse import ArgumentParser
from typing import Sequence

from .config import load_config, apply_env_overrides, set_env_from_config

# Load configuration at startup so modules relying on environment variables
# pick up the values from config files or environment.
_cfg = apply_env_overrides(load_config())
set_env_from_config(_cfg)


from .scanner import scan_tokens_async
from .onchain_metrics import top_volume_tokens
from .market_ws import listen_and_trade
from .simulation import run_simulations
from .decision import should_buy, should_sell
from .prices import fetch_token_prices_async

from .memory import Memory
from .portfolio import Portfolio
from .exchange import place_order_async
from .strategy_manager import StrategyManager

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
    trailing_stop: float | None = None,
    max_drawdown: float = 1.0,
    volatility_factor: float = 1.0,
    arbitrage_threshold: float | None = None,
    arbitrage_amount: float | None = None,
    strategy_manager: StrategyManager | None = None,
) -> None:
    """Execute a single trading iteration asynchronously."""

    if arbitrage_threshold is None:
        arbitrage_threshold = float(os.getenv("ARBITRAGE_THRESHOLD", "0") or 0)
    if arbitrage_amount is None:
        arbitrage_amount = float(os.getenv("ARBITRAGE_AMOUNT", "0") or 0)


    
    scan_kwargs = {"offline": offline, "token_file": token_file}
    if discovery_method != "websocket":
        scan_kwargs["method"] = discovery_method

    tokens = await scan_tokens_async(**scan_kwargs)


    try:
        tokens = await scan_tokens_async(
            offline=offline, token_file=token_file, method=discovery_method
        )
    except TypeError:
        # Support tests that monkeypatch ``scan_tokens_async`` without the
        # ``method`` parameter.

        tokens = await scan_tokens_async(offline=offline, token_file=token_file)

    # Always consider existing holdings when making sell decisions
    tokens = list(set(tokens) | set(portfolio.balances.keys()))

    rpc_url = os.getenv("SOLANA_RPC_URL")
    if rpc_url and not offline:
        try:

            ranked = top_volume_tokens(rpc_url, limit=len(tokens))
            ranked_set = set(ranked)
            tokens = [t for t in ranked if t in tokens] + [t for t in tokens if t not in ranked_set]

        except Exception as exc:  # pragma: no cover - network errors
            logging.warning("Volume ranking failed: %s", exc)


    if strategy_manager is None:
        strategy_manager = StrategyManager()

    for token in tokens:
        try:
            actions = await strategy_manager.evaluate(token, portfolio)
        except Exception as exc:  # pragma: no cover - strategy errors
            logging.warning("Strategy evaluation failed for %s: %s", token, exc)
            continue
        for action in actions:
            side = action.get("side")
            amount = action.get("amount", 0.0)
            price = action.get("price", 0.0)
            if side not in {"buy", "sell"} or amount <= 0:
                continue

            await place_order_async(
                token,
                side=side,
                amount=amount,
                price=price,
                testnet=testnet,
                dry_run=dry_run,
                keypair=keypair,
            )

            if not dry_run:
                memory.log_trade(token=token, direction=side, amount=amount, price=price)
                portfolio.update(token, amount if side == "buy" else -amount, price)




def main(
    memory_path: str = "sqlite:///memory.db",
    loop_delay: int = 60,
    *,
    iterations: int | None = None,
    testnet: bool = False,
    dry_run: bool = False,
    offline: bool = False,


    token_file: str | None = None,
    discovery_method: str | None = None,


    keypair_path: str | None = None,
    portfolio_path: str = "portfolio.json",
    config_path: str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    max_drawdown: float | None = None,
    volatility_factor: float | None = None,
    risk_tolerance: float | None = None,
    max_allocation: float | None = None,
    risk_multiplier: float | None = None,
    market_ws_url: str | None = None,
    arbitrage_threshold: float | None = None,
    arbitrage_amount: float | None = None,
    arbitrage_tokens: list[str] | None = None,
    strategies: list[str] | None = None,

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

    discovery_method:
        Token discovery method: onchain, websocket, mempool, pools or file.

    portfolio_path:
        Path to the JSON file for persisting portfolio state.

    strategies:
        Optional list of strategy module names to load.





    """

    from .wallet import load_keypair

    cfg = apply_env_overrides(load_config(config_path))
    set_env_from_config(cfg)

    if risk_tolerance is not None:
        os.environ["RISK_TOLERANCE"] = str(risk_tolerance)
    if max_allocation is not None:
        os.environ["MAX_ALLOCATION"] = str(max_allocation)
    if risk_multiplier is not None:
        os.environ["RISK_MULTIPLIER"] = str(risk_multiplier)

    if discovery_method is None:
        discovery_method = cfg.get("discovery_method")
    if discovery_method is None:
        discovery_method = os.getenv("DISCOVERY_METHOD", "websocket")

    if stop_loss is None:
        stop_loss = cfg.get("stop_loss")
    if take_profit is None:
        take_profit = cfg.get("take_profit")
    if trailing_stop is None:
        trailing_stop = cfg.get("trailing_stop")
    if max_drawdown is None:
        max_drawdown = float(cfg.get("max_drawdown", 1.0))
    if volatility_factor is None:
        volatility_factor = float(cfg.get("volatility_factor", 1.0))
    if arbitrage_threshold is None:
        arbitrage_threshold = float(cfg.get("arbitrage_threshold", 0.0))
    if arbitrage_amount is None:
        arbitrage_amount = float(cfg.get("arbitrage_amount", 0.0))
    if strategies is None:
        strategies = cfg.get("strategies")
        if isinstance(strategies, str):
            strategies = [s.strip() for s in strategies.split(",") if s.strip()]
    if market_ws_url is None:
        market_ws_url = cfg.get("market_ws_url")
    if market_ws_url is None:
        market_ws_url = os.getenv("MARKET_WS_URL")
    if arbitrage_tokens is None:
        tokens_cfg = cfg.get("arbitrage_tokens")
        if isinstance(tokens_cfg, str):
            arbitrage_tokens = [t.strip() for t in tokens_cfg.split(",") if t.strip()]
        elif tokens_cfg:
            arbitrage_tokens = list(tokens_cfg)
    if arbitrage_tokens is None:
        env_tokens = os.getenv("ARBITRAGE_TOKENS")
        if env_tokens:
            arbitrage_tokens = [t.strip() for t in env_tokens.split(",") if t.strip()]

    memory = Memory(memory_path)
    portfolio = Portfolio(path=portfolio_path)

    strategy_manager = StrategyManager(strategies)

    keypair = load_keypair(keypair_path) if keypair_path else None

    async def loop() -> None:
        ws_task = None
        arb_task = None
        if market_ws_url:
            ws_task = asyncio.create_task(
                listen_and_trade(
                    market_ws_url,
                    memory,
                    portfolio,
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )
            )

        if arbitrage_tokens:
            async def monitor_arbitrage() -> None:
                while True:
                    try:
                        await arbitrage.detect_and_execute_arbitrage(
                            arbitrage_tokens,
                            threshold=arbitrage_threshold,
                            amount=arbitrage_amount,
                            testnet=testnet,
                            dry_run=dry_run,
                            keypair=keypair,
                        )
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.warning("Arbitrage monitor failed: %s", exc)
                    await asyncio.sleep(loop_delay)

            arb_task = asyncio.create_task(monitor_arbitrage())

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
                    trailing_stop=trailing_stop,
                    max_drawdown=max_drawdown,
                    volatility_factor=volatility_factor,
                    arbitrage_threshold=arbitrage_threshold,
                    arbitrage_amount=arbitrage_amount,
                    strategy_manager=strategy_manager,
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
                    trailing_stop=trailing_stop,
                    max_drawdown=max_drawdown,
                    volatility_factor=volatility_factor,
                    arbitrage_threshold=arbitrage_threshold,
                    arbitrage_amount=arbitrage_amount,
                    strategy_manager=strategy_manager,
                )
                if i < iterations - 1:
                    await asyncio.sleep(loop_delay)

        if ws_task:
            ws_task.cancel()
            with contextlib.suppress(Exception):
                await ws_task
        if arb_task:
            arb_task.cancel()
            with contextlib.suppress(Exception):
                await arb_task

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
        "--token-list",
        dest="token_file",
        help="Load token addresses from FILE (one per line)",
    )
    parser.add_argument(
        "--discovery-method",

        default=None,
        choices=["websocket", "onchain", "mempool", "pools", "file"],
        help="Token discovery method",
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
        help="Stop loss threshold as a fraction (e.g. 0.1 for 10%%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit threshold as a fraction",
    )
    parser.add_argument(
        "--trailing-stop",
        type=float,
        default=None,
        help="Trailing stop percentage",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Maximum allowed portfolio drawdown",
    )
    parser.add_argument(
        "--volatility-factor",
        type=float,
        default=None,
        help="Scaling factor for volatility in position sizing",
    )
    parser.add_argument(
        "--risk-tolerance",
        type=float,
        default=None,
        help="Base risk tolerance for position sizing",
    )
    parser.add_argument(
        "--max-allocation",
        type=float,
        default=None,
        help="Maximum portfolio allocation per trade",
    )
    parser.add_argument(
        "--risk-multiplier",
        type=float,
        default=None,
        help="Multiplier applied to risk parameters",
    )
    parser.add_argument(
        "--market-ws-url",
        default=None,
        help="Websocket URL for real-time market events",
    )
    parser.add_argument(
        "--arbitrage-threshold",
        type=float,
        default=None,
        help="Minimum price diff fraction for arbitrage",
    )
    parser.add_argument(
        "--arbitrage-amount",
        type=float,
        default=None,
        help="Trade size when executing arbitrage",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated list of strategy modules",

    )
    args = parser.parse_args()
    main(
        memory_path=args.memory_path,
        loop_delay=args.loop_delay,
        iterations=args.iterations,
        testnet=args.testnet,
        dry_run=args.dry_run,
        offline=args.offline,


        token_file=args.token_file,

        discovery_method=args.discovery_method,

        keypair_path=args.keypair,
        portfolio_path=args.portfolio_path,
        config_path=args.config,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        trailing_stop=args.trailing_stop,
        max_drawdown=args.max_drawdown,
        volatility_factor=args.volatility_factor,
        risk_tolerance=args.risk_tolerance,
        max_allocation=args.max_allocation,
        risk_multiplier=args.risk_multiplier,
        market_ws_url=args.market_ws_url,
        arbitrage_threshold=args.arbitrage_threshold,
        arbitrage_amount=args.arbitrage_amount,
        arbitrage_tokens=None,
        strategies=[s.strip() for s in args.strategies.split(',')] if args.strategies else None,

    )
