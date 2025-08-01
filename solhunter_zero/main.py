import logging
import os
import asyncio
import sys
import contextlib
import subprocess
import time
import psutil
from argparse import ArgumentParser
import cProfile
from typing import Sequence

from .util import install_uvloop

from pathlib import Path

from .config import (
    load_config,
    apply_env_overrides,
    set_env_from_config,
    load_selected_config,
    get_active_config_name,
    get_event_bus_url,
    CONFIG_DIR,
)
from .http import close_session
from . import wallet

_SERVICE_MANIFEST = (
    Path(__file__).resolve().parent.parent / "depth_service" / "Cargo.toml"
)

install_uvloop()


def _start_depth_service(cfg: dict) -> subprocess.Popen | None:
    """Launch the Rust depth_service if enabled."""

    if not cfg.get("depth_service"):
        return None

    args = [
        "cargo",
        "run",
        "--manifest-path",
        str(_SERVICE_MANIFEST),
        "--release",
        "--",
    ]

    def add(flag: str, key: str) -> None:
        val = os.getenv(key.upper()) or cfg.get(key)
        if val:
            args.extend([flag, str(val)])

    add("--raydium", "raydium_ws_url")
    add("--orca", "orca_ws_url")
    add("--phoenix", "phoenix_ws_url")
    add("--meteora", "meteora_ws_url")
    add("--jupiter", "jupiter_ws_url")
    add("--serum", "serum_ws_url")

    rpc = os.getenv("SOLANA_RPC_URL") or cfg.get("solana_rpc_url")
    if rpc:
        args.extend(["--rpc", rpc])
    keypair = os.getenv("SOLANA_KEYPAIR") or os.getenv("KEYPAIR_PATH")
    if keypair:
        args.extend(["--keypair", keypair])

    proc = subprocess.Popen(args)

    socket_path = os.getenv("DEPTH_SERVICE_SOCKET", "/tmp/depth_service.sock")

    async def wait_for_socket() -> None:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(socket_path)
            except Exception:
                if time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.05)
            else:
                writer.close()
                await writer.wait_closed()
                return

    asyncio.run(wait_for_socket())
    return proc


# Load configuration at startup so modules relying on environment variables
# pick up the values from config files or environment.
_cfg = apply_env_overrides(load_config())
set_env_from_config(_cfg)


from .token_scanner import scan_tokens_async
from .onchain_metrics import async_top_volume_tokens, fetch_dex_metrics_async
from .market_ws import listen_and_trade
from .simulation import run_simulations
from .decision import should_buy, should_sell
from .prices import fetch_token_prices_async

from . import order_book_ws

from .memory import Memory
from .portfolio import Portfolio
from .exchange import place_order_async
from .strategy_manager import StrategyManager
from .agent_manager import AgentManager
from .agents.discovery import DiscoveryAgent

from .portfolio import dynamic_order_size
from .agents.conviction import predict_price_movement
from .risk import RiskManager, recent_value_at_risk
from . import arbitrage
from . import depth_client
from . import event_bus
from .data_pipeline import start_depth_snapshot_listener

# keep track of recently traded tokens for scheduling
_LAST_TOKENS: list[str] = []

_HIGH_RISK_PRESET = Path(__file__).resolve().parent.parent / "config.highrisk.toml"

_level_name = os.getenv("LOG_LEVEL") or str(_cfg.get("log_level", "INFO"))
logging.basicConfig(level=getattr(logging, _level_name.upper(), logging.INFO))


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
    agent_manager: AgentManager | None = None,
) -> None:
    """Execute a single trading iteration asynchronously."""

    if arbitrage_threshold is None:
        arbitrage_threshold = float(os.getenv("ARBITRAGE_THRESHOLD", "0") or 0)
    if arbitrage_amount is None:
        arbitrage_amount = float(os.getenv("ARBITRAGE_AMOUNT", "0") or 0)

    scan_kwargs = {
        "offline": offline,
        "token_file": token_file,
        "method": discovery_method,
    }

    if agent_manager is not None:
        if hasattr(agent_manager, "discover_tokens"):
            try:
                tokens = await agent_manager.discover_tokens(**scan_kwargs)
            except TypeError:
                tokens = await agent_manager.discover_tokens(
                    offline=offline, token_file=token_file
                )
        else:
            tokens = await scan_tokens_async(dynamic_concurrency=True, **scan_kwargs)
    else:
        tokens = await DiscoveryAgent().discover_tokens(**scan_kwargs)

    global _LAST_TOKENS
    _LAST_TOKENS = list(tokens)

    # Always consider existing holdings when making sell decisions
    tokens = list(set(tokens) | set(portfolio.balances.keys()))

    rpc_url = os.getenv("SOLANA_RPC_URL")
    if rpc_url and not offline:
        try:
            ranked = await async_top_volume_tokens(rpc_url, limit=len(tokens))
            ranked_set = set(ranked)
            tokens = [t for t in ranked if t in tokens] + [
                t for t in tokens if t not in ranked_set
            ]

        except Exception as exc:  # pragma: no cover - network errors
            logging.warning("Volume ranking failed: %s", exc)

    price_lookup = {}
    if portfolio.balances:
        if not offline:
            price_lookup = await fetch_token_prices_async(portfolio.balances.keys())
        portfolio.update_drawdown(price_lookup)
        if price_lookup:
            portfolio.record_prices(price_lookup)
            portfolio.update_risk_metrics()
    drawdown = portfolio.current_drawdown(price_lookup)
    risk_metrics = portfolio.risk_metrics
    os.environ["PORTFOLIO_VALUE"] = str(portfolio.total_value(price_lookup))

    if agent_manager is not None:
        for token in tokens:
            try:
                await agent_manager.execute(token, portfolio)
            except Exception as exc:  # pragma: no cover - agent errors
                logging.warning("Agent execution failed for %s: %s", token, exc)
        return

    use_old = (
        strategy_manager is None
        and run_simulations.__module__ != "solhunter_zero.simulation"
    )
    if use_old:
        for token in tokens:
            sims = run_simulations(token, count=100)

            if arbitrage_amount > 0 and arbitrage_threshold > 0:
                try:
                    await arbitrage.detect_and_execute_arbitrage(
                        token,
                        threshold=arbitrage_threshold,
                        amount=arbitrage_amount,
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    logging.warning("Arbitrage check failed: %s", exc)

            if should_buy(sims):
                logging.info("Buying %s", token)
                avg_roi = sum(r.expected_roi for r in sims) / len(sims)
                if price_lookup:
                    balance = portfolio.total_value(price_lookup)
                    alloc = portfolio.percent_allocated(token, price_lookup)
                else:
                    balance = sum(p.amount for p in portfolio.balances.values()) or 1.0
                    alloc = portfolio.percent_allocated(token)

                rm = RiskManager.from_config(
                    {
                        "risk_tolerance": os.getenv("RISK_TOLERANCE", "0.1"),
                        "max_allocation": os.getenv("MAX_ALLOCATION", "0.2"),
                        "max_risk_per_token": os.getenv("MAX_RISK_PER_TOKEN", "0.1"),
                        "max_drawdown": max_drawdown,
                        "volatility_factor": volatility_factor,
                        "risk_multiplier": os.getenv("RISK_MULTIPLIER", "1.0"),
                        "min_portfolio_value": os.getenv("MIN_PORTFOLIO_VALUE", "20"),
                    }
                )

                first_sim = sims[0] if sims else None
                from .risk import hedge_ratio, leverage_scaling

                hedge = hedge_ratio(
                    portfolio.price_history.get(token, []),
                    portfolio.price_history.get("USDC", []),
                )
                lev = leverage_scaling(1.0, 1.0 / (1 + abs(hedge)))
                var_conf = float(os.getenv("VAR_CONFIDENCE", "0.95"))
                var_window = int(os.getenv("VAR_WINDOW", "30"))
                var_threshold = float(os.getenv("VAR_THRESHOLD", "0"))
                hist = portfolio.price_history.get(token, [])
                var = recent_value_at_risk(hist, window=var_window, confidence=var_conf)
                params = rm.adjusted(
                    drawdown,
                    0.0,
                    volume_spike=getattr(first_sim, "volume_spike", 1.0),
                    depth_change=getattr(first_sim, "depth_change", 0.0),
                    whale_activity=getattr(first_sim, "whale_activity", 0.0),
                    portfolio_value=balance,
                    portfolio_metrics=risk_metrics,
                    leverage=lev,
                    correlation=risk_metrics.get("correlation"),
                    prices=hist,
                    var_threshold=var_threshold,
                    var_confidence=var_conf,
                )

                try:
                    pred_roi = predict_price_movement(token)
                except Exception:
                    pred_roi = 0.0

                amount = dynamic_order_size(
                    balance,
                    avg_roi,
                    pred_roi,
                    0.0,
                    0.0,
                    risk_tolerance=params.risk_tolerance,
                    max_allocation=params.max_allocation,
                    max_risk_per_token=params.max_risk_per_token,
                    max_drawdown=max_drawdown,
                    volatility_factor=volatility_factor,
                    current_allocation=alloc,
                    min_portfolio_value=params.min_portfolio_value,
                    correlation=risk_metrics.get("correlation"),
                    var=var,
                    var_threshold=var_threshold,
                )
                await place_order_async(
                    token,
                    side="buy",
                    amount=amount,
                    price=0,
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )

                if not dry_run:
                    await memory.log_trade(
                        token=token, direction="buy", amount=amount, price=0
                    )
                    await portfolio.update_async(token, amount, 0)

        price_lookup_sell = {}
        if (
            stop_loss is not None
            or take_profit is not None
            or trailing_stop is not None
        ):
            price_lookup_sell = await fetch_token_prices_async(
                portfolio.balances.keys()
            )
            await portfolio.update_highs_async(price_lookup_sell)
            if price_lookup_sell:
                portfolio.record_prices(price_lookup_sell)
                portfolio.update_risk_metrics()

        for token, pos in list(portfolio.balances.items()):
            sims = run_simulations(token, count=100)

            roi_trigger = False
            if token in price_lookup_sell:
                price = price_lookup_sell[token]
                roi = portfolio.position_roi(token, price)
                if stop_loss is not None and roi <= -stop_loss:
                    roi_trigger = True
                if take_profit is not None and roi >= take_profit:
                    roi_trigger = True
                if trailing_stop is not None and portfolio.trailing_stop_triggered(
                    token, price, trailing_stop
                ):
                    roi_trigger = True

            if roi_trigger or should_sell(
                sims,
                trailing_stop=trailing_stop,
                current_price=price_lookup_sell.get(token),
                high_price=pos.high_price,
            ):
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
                    await memory.log_trade(
                        token=token, direction="sell", amount=pos.amount, price=0
                    )
                    await portfolio.update_async(token, -pos.amount, 0)
    else:
        if strategy_manager is None:
            strategy_manager = StrategyManager()

        for token in tokens:
            if arbitrage_amount > 0 and arbitrage_threshold > 0:
                try:
                    await arbitrage.detect_and_execute_arbitrage(
                        token,
                        threshold=arbitrage_threshold,
                        amount=arbitrage_amount,
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                except Exception as exc:  # pragma: no cover - network errors
                    logging.warning("Arbitrage check failed: %s", exc)
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
                    await memory.log_trade(
                        token=token, direction=side, amount=amount, price=price
                    )
                    await portfolio.update_async(
                        token,
                        amount if side == "buy" else -amount,
                        price,
                    )

        if agent_manager is not None:
            agent_manager.update_weights()
            agent_manager.save_weights()


async def _init_rl_training(
    cfg: dict,
    *,
    rl_daemon: bool = False,
    rl_interval: float = 3600.0,
) -> asyncio.Task | None:
    """Set up RL background training if enabled."""

    auto_train_cfg = bool(cfg.get("rl_auto_train", False))
    if not rl_daemon and not auto_train_cfg:
        return None

    from .rl_daemon import RLDaemon
    from .event_bus import subscription
    import torch

    mem_db = cfg.get("memory_path", "sqlite:///memory.db")
    data_path = cfg.get("rl_db_path", "offline_data.db")
    model_path = cfg.get("rl_model_path", "ppo_model.pt")
    algo = cfg.get("rl_algo", "ppo")
    policy = cfg.get("rl_policy", "mlp")
    auto_train = auto_train_cfg
    tune_interval = float(cfg.get("rl_tune_interval", rl_interval))
    cpu_count = os.cpu_count() or 1
    dyn_workers = bool(cfg.get("rl_dynamic_workers", cpu_count > 1))

    daemon = RLDaemon(
        memory_path=mem_db,
        data_path=data_path,
        model_path=model_path,
        algo=algo,
        policy=policy,
        dynamic_workers=dyn_workers,
    )
    task = daemon.start(rl_interval, auto_train=auto_train, tune_interval=tune_interval)

    def _reload(_payload):
        try:
            daemon.model.load_state_dict(
                torch.load(model_path, map_location=daemon.device)
            )
        except Exception:
            return
        for ag in daemon.agents:
            try:
                if hasattr(ag, "reload_weights"):
                    ag.reload_weights()
                else:
                    ag._load_weights()
            except Exception:
                continue

    sub = subscription("rl_weights", _reload)
    sub.__enter__()

    return task


def main(
    memory_path: str = "sqlite:///memory.db",
    loop_delay: int = 60,
    min_delay: int | None = None,
    max_delay: int | None = None,
    cpu_low_threshold: float | None = None,
    cpu_high_threshold: float | None = None,
    depth_freq_low: float | None = None,
    depth_freq_high: float | None = None,
    depth_rate_limit: float = 0.1,
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
    order_book_ws_url: str | None = None,
    arbitrage_threshold: float | None = None,
    arbitrage_amount: float | None = None,
    arbitrage_tokens: list[str] | None = None,
    strategies: list[str] | None = None,
    rl_daemon: bool = False,
    rl_interval: float = 3600.0,
    dynamic_concurrency: bool = False,
    strategy_rotation_interval: int | None = None,
    weight_config_paths: list[str] | None = None,
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

    arbitrage_tokens:
        Optional list of token addresses to monitor for arbitrage opportunities.

    strategies:
        Optional list of strategy module names to load.





    """

    from .wallet import load_keypair

    cfg = apply_env_overrides(load_config(config_path))
    prev_agents = os.environ.get("AGENTS")
    prev_weights = os.environ.get("AGENT_WEIGHTS")
    set_env_from_config(cfg)

    use_bundles = str(
        cfg.get("use_mev_bundles") or os.getenv("USE_MEV_BUNDLES", "false")
    ).lower() in {"1", "true", "yes"}
    if use_bundles and (not os.getenv("JITO_RPC_URL") or not os.getenv("JITO_AUTH")):
        logging.warning("MEV bundles enabled but JITO_RPC_URL or JITO_AUTH is missing")

    proc = _start_depth_service(cfg)

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
    if order_book_ws_url is None:
        order_book_ws_url = cfg.get("order_book_ws_url")
    if order_book_ws_url is None:
        order_book_ws_url = os.getenv("ORDER_BOOK_WS_URL")
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

    if min_delay is None:
        min_delay = int(cfg.get("min_delay", 1))
    if max_delay is None:
        max_delay = int(cfg.get("max_delay", loop_delay))
    if cpu_low_threshold is None:
        cpu_low_threshold = float(cfg.get("cpu_low_threshold", 20.0))
    if cpu_high_threshold is None:
        cpu_high_threshold = float(cfg.get("cpu_high_threshold", 80.0))
    if depth_freq_low is None:
        depth_freq_low = float(cfg.get("depth_freq_low", 1.0))
    if depth_freq_high is None:
        depth_freq_high = float(cfg.get("depth_freq_high", 10.0))

    if strategy_rotation_interval is None:
        strategy_rotation_interval = int(cfg.get("strategy_rotation_interval", 0))
    if weight_config_paths is None:
        paths = cfg.get("weight_config_paths") or []
        if isinstance(paths, str):
            weight_config_paths = [p.strip() for p in paths.split(",") if p.strip()]
        else:
            weight_config_paths = list(paths) if paths else []

    memory = Memory(memory_path)
    portfolio = Portfolio(path=portfolio_path)

    agent_manager: AgentManager | None = None

    if cfg.get("agents"):
        if weight_config_paths:
            cfg["weight_config_paths"] = weight_config_paths
        if strategy_rotation_interval is not None:
            cfg["strategy_rotation_interval"] = strategy_rotation_interval
        agent_manager = AgentManager.from_config(cfg)
        if agent_manager is None:
            strategy_manager = StrategyManager(strategies)
        else:
            strategy_manager = None

    else:
        strategy_manager = StrategyManager(strategies)

    if keypair_path:
        keypair = load_keypair(keypair_path)
    else:
        keypair = wallet.load_selected_keypair()

    async def loop() -> None:
        ws_task = None
        book_task = None
        arb_task = None
        depth_task = None
        depth_updates = 0
        prev_count = 0
        prev_ts = time.monotonic()
        nonlocal depth_rate_limit
        def _count(_p):
            nonlocal depth_updates
            depth_updates += 1
        unsub_counter = event_bus.subscribe("depth_update", _count)
        bus_started = False
        if get_event_bus_url() is None:
            await event_bus.start_ws_server()
            bus_started = True

        rl_task = await _init_rl_training(
            cfg, rl_daemon=rl_daemon, rl_interval=rl_interval
        )
        collect_data = str(
            cfg.get("collect_offline_data")
            or os.getenv("COLLECT_OFFLINE_DATA", "false")
        ).lower() in {"1", "true", "yes"}
        stop_collector = None
        if collect_data:
            db_path = cfg.get("rl_db_path", "offline_data.db")
            stop_collector = start_depth_snapshot_listener(db_path)
        prev_activity = 0.0
        iteration_idx = 0

        def adjust_delay(metrics: dict) -> None:
            nonlocal loop_delay, prev_activity, prev_count, prev_ts, depth_rate_limit
            activity = metrics.get("liquidity", 0.0) + metrics.get("volume", 0.0)
            cpu = psutil.cpu_percent()
            now = time.monotonic()
            freq = (depth_updates - prev_count) / (now - prev_ts) if now > prev_ts else 0.0
            if (
                activity > prev_activity * 1.5
                or freq > depth_freq_high
                or cpu > cpu_high_threshold
            ):
                loop_delay = max(min_delay, max(1, loop_delay // 2))
                depth_rate_limit = max(0.01, depth_rate_limit / 2)
            elif (
                activity < prev_activity * 0.5
                and freq < depth_freq_low
                and cpu < cpu_low_threshold
            ):
                loop_delay = min(max_delay, loop_delay * 2)
                depth_rate_limit = min(1.0, depth_rate_limit * 2)
            prev_activity = activity
            prev_count = depth_updates
            prev_ts = now
            arbitrage.DEPTH_RATE_LIMIT = depth_rate_limit

        if market_ws_url:

            async def run_market_ws() -> None:
                while True:
                    try:
                        await listen_and_trade(
                            market_ws_url,
                            memory,
                            portfolio,
                            testnet=testnet,
                            dry_run=dry_run,
                            keypair=keypair,
                        )
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Market websocket failed: %s", exc)
                        await asyncio.sleep(1.0)

            ws_task = asyncio.create_task(run_market_ws())

        use_depth_stream = os.getenv("USE_DEPTH_STREAM", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_depth_stream:

            async def run_depth_ws() -> None:
                while True:
                    try:
                        await depth_client.listen_depth_ws()
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Depth websocket failed: %s", exc)
                        await asyncio.sleep(1.0)

            depth_task = asyncio.create_task(run_depth_ws())

        if order_book_ws_url:

            async def run_order_book() -> None:
                while True:
                    try:
                        async for _ in order_book_ws.stream_order_book(
                            order_book_ws_url,
                            rate_limit=depth_rate_limit
                        ):
                            pass
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Order book stream failed: %s", exc)
                        await asyncio.sleep(1.0)

            book_task = asyncio.create_task(run_order_book())

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
                    agent_manager=agent_manager,
                )
                if _LAST_TOKENS:
                    metrics = await fetch_dex_metrics_async(
                        _LAST_TOKENS[0], os.getenv("METRICS_BASE_URL")
                    )
                    adjust_delay(metrics)
                iteration_idx += 1
                if (
                    agent_manager
                    and iteration_idx % getattr(agent_manager, "evolve_interval", 1)
                    == 0
                ):
                    agent_manager.evolve(
                        threshold=getattr(agent_manager, "mutation_threshold", 0.0)
                    )
                if (
                    agent_manager
                    and getattr(agent_manager, "strategy_rotation_interval", 0) > 0
                    and iteration_idx % agent_manager.strategy_rotation_interval == 0
                ):
                    agent_manager.rotate_weight_configs()
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
                    agent_manager=agent_manager,
                )
                if _LAST_TOKENS:
                    metrics = await fetch_dex_metrics_async(
                        _LAST_TOKENS[0], os.getenv("METRICS_BASE_URL")
                    )
                    adjust_delay(metrics)
                iteration_idx += 1
                if (
                    agent_manager
                    and iteration_idx % getattr(agent_manager, "evolve_interval", 1)
                    == 0
                ):
                    agent_manager.evolve(
                        threshold=getattr(agent_manager, "mutation_threshold", 0.0)
                    )
                if (
                    agent_manager
                    and getattr(agent_manager, "strategy_rotation_interval", 0) > 0
                    and iteration_idx % agent_manager.strategy_rotation_interval == 0
                ):
                    agent_manager.rotate_weight_configs()
                if i < iterations - 1:
                    await asyncio.sleep(loop_delay)

        if ws_task:
            ws_task.cancel()
            with contextlib.suppress(Exception):
                await ws_task
        if book_task:
            book_task.cancel()
            with contextlib.suppress(Exception):
                await book_task
        if arb_task:
            arb_task.cancel()
            with contextlib.suppress(Exception):
                await arb_task
        if depth_task:
            depth_task.cancel()
            with contextlib.suppress(Exception):
                await depth_task
        if rl_task:
            rl_task.cancel()
            with contextlib.suppress(Exception):
                await rl_task
        if stop_collector:
            stop_collector()
        if bus_started:
            await event_bus.stop_ws_server()
        unsub_counter()

    try:
        asyncio.run(loop())
    finally:
        if prev_agents is None:
            os.environ.pop("AGENTS", None)
        else:
            os.environ["AGENTS"] = prev_agents
        if prev_weights is None:
            os.environ.pop("AGENT_WEIGHTS", None)
        else:
            os.environ["AGENT_WEIGHTS"] = prev_weights


def run_auto(**kwargs) -> None:
    """Start trading with selected config or high-risk preset."""
    cfg = load_selected_config()
    cfg_path = None
    if cfg:
        name = get_active_config_name()
        cfg_path = os.path.join(CONFIG_DIR, name) if name else None
    elif _HIGH_RISK_PRESET.is_file():
        cfg_path = str(_HIGH_RISK_PRESET)
        cfg = load_config(cfg_path)
    cfg = apply_env_overrides(cfg)
    prev_agents = os.environ.get("AGENTS")
    prev_weights = os.environ.get("AGENT_WEIGHTS")
    set_env_from_config(cfg)

    try:
        from . import data_sync

        asyncio.run(data_sync.sync_recent())
    except Exception as exc:  # pragma: no cover - ignore sync errors
        logging.getLogger(__name__).warning("data sync failed: %s", exc)

    active_name = wallet.get_active_keypair_name()
    if active_name is None:
        keys = wallet.list_keypairs()
        if len(keys) == 1:
            wallet.select_keypair(keys[0])
            active_name = keys[0]

    if active_name and not os.getenv("KEYPAIR_PATH"):
        os.environ["KEYPAIR_PATH"] = os.path.join(
            wallet.KEYPAIR_DIR, active_name + ".json"
        )

    try:
        main(config_path=cfg_path, **kwargs)
    finally:
        if prev_agents is None:
            os.environ.pop("AGENTS", None)
        else:
            os.environ["AGENTS"] = prev_agents
        if prev_weights is None:
            os.environ.pop("AGENT_WEIGHTS", None)
        else:
            os.environ["AGENT_WEIGHTS"] = prev_weights


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
        "--order-book-ws-url",
        default=None,
        help="Websocket URL for order book depth updates",
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
    parser.add_argument(
        "--rl-daemon", action="store_true", help="Start RL training daemon"
    )
    parser.add_argument(
        "--rl-interval",
        type=float,
        default=3600.0,
        help="Seconds between RL training cycles",
    )
    parser.add_argument(
        "--dynamic-concurrency",
        action="store_true",
        help="Dynamically adjust ranking concurrency based on CPU usage",
    )
    parser.add_argument(
        "--strategy-rotation-interval",
        type=int,
        default=0,
        help="Iterations between weight config evaluations",
    )
    parser.add_argument(
        "--weight-config",
        dest="weight_configs",
        action="append",
        default=[],
        help="Configuration file with agent_weights to rotate",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the trading loop and write stats to profile.out",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Load selected config and start trading automatically",
    )
    args = parser.parse_args()
    kwargs = dict(
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
        order_book_ws_url=args.order_book_ws_url,
        arbitrage_threshold=args.arbitrage_threshold,
        arbitrage_amount=args.arbitrage_amount,
        arbitrage_tokens=None,
        strategies=(
            [s.strip() for s in args.strategies.split(",")] if args.strategies else None
        ),
        min_delay=None,
        max_delay=None,
        cpu_low_threshold=None,
        cpu_high_threshold=None,
        depth_freq_low=None,
        depth_freq_high=None,
        depth_rate_limit=0.1,
        rl_daemon=args.rl_daemon,
        rl_interval=args.rl_interval,
        dynamic_concurrency=args.dynamic_concurrency,
        strategy_rotation_interval=args.strategy_rotation_interval,
        weight_config_paths=args.weight_configs,
    )

    try:
        if args.profile:
            if args.auto:
                cProfile.runctx(
                    "run_auto(**kwargs)", globals(), locals(), filename="profile.out"
                )
            else:
                cProfile.runctx(
                    "main(**kwargs)", globals(), locals(), filename="profile.out"
                )
        else:
            if args.auto:
                run_auto(**kwargs)
            else:
                main(**kwargs)
    finally:
        asyncio.run(close_session())
