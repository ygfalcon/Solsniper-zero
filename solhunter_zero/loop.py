"""Trading loop implementation for SolHunter Zero."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Sequence

from . import metrics_aggregator
from .exchange import place_order_async as _exchange_place_order_async
from .paths import ROOT
from .services import depth_service_watchdog
from .system import detect_cpu_count
from .main_state import TradingState
from .util import parse_bool_env


_PROCESS_START_TIME = time.perf_counter()
_APP_START_TIME = time.monotonic()

_first_trade_recorded = False
_first_trade_event = asyncio.Event()


class FirstTradeTimeoutError(RuntimeError):
    """Raised when no trade occurs before the configured timeout."""

    pass


async def place_order_async(*args, **kwargs):
    """Emit time-to-first-trade metric on first successful order."""
    global _first_trade_recorded
    result = await _exchange_place_order_async(*args, **kwargs)
    if not _first_trade_recorded:
        _first_trade_recorded = True
        _first_trade_event.set()
        try:
            metrics_aggregator.publish(
                "time_to_first_trade", time.monotonic() - _APP_START_TIME
            )
        except Exception:  # pragma: no cover - metric errors
            logging.exception("Failed to publish time_to_first_trade metric")
    return result


async def _check_first_trade(timeout: float, retry: bool) -> None:
    """Wait for first trade or timeout and optionally request retry."""
    try:
        await asyncio.wait_for(_first_trade_event.wait(), timeout)
    except asyncio.TimeoutError:
        logging.error("First trade not recorded within %s seconds", timeout)
        if retry:
            raise FirstTradeTimeoutError


async def run_iteration(
    memory,
    portfolio,
    state: TradingState,
    cfg=None,
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
    strategy_manager=None,
    agent_manager=None,
) -> None:
    """Execute a single trading iteration asynchronously."""

    from .token_scanner import scan_tokens_async
    from .agents.discovery import DiscoveryAgent
    from .onchain_metrics import async_top_volume_tokens
    from .simulation import run_simulations
    from .decision import should_buy, should_sell
    from .prices import fetch_token_prices_async
    from .portfolio import dynamic_order_size
    from .agents.conviction import predict_price_movement
    from .risk import RiskManager, recent_value_at_risk

    from .memory import Memory  # type: ignore
    from .portfolio import Portfolio  # type: ignore
    from .config_runtime import Config  # type: ignore

    await memory.wait_ready()
    metrics_aggregator.start()

    if cfg is None:
        cfg = Config.from_env({})  # type: ignore[arg-type]

    if arbitrage_threshold is None:
        arbitrage_threshold = cfg.arbitrage_threshold
    if arbitrage_amount is None:
        arbitrage_amount = cfg.arbitrage_amount

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

    state.last_tokens = list(tokens)

    tokens = list(set(tokens) | set(portfolio.balances.keys()))

    recent_window = cfg.recent_trade_window
    if recent_window > 0:
        now = datetime.datetime.utcnow()
        tokens = [
            t
            for t in tokens
            if (
                (ts := state.last_trade_times.get(t)) is None
                or (now - ts).total_seconds() > recent_window
            )
        ]

    rpc_url = cfg.solana_rpc_url
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
            if should_buy(sims):
                logging.info("Buying %s", token)
                avg_roi = sum(r.expected_roi for r in sims) / len(sims)
                if price_lookup:
                    balance = portfolio.total_value(price_lookup)
                    alloc = portfolio.percent_allocated(token, price_lookup)
                else:
                    balance = 0
                    alloc = 0
                amount = dynamic_order_size(
                    balance,
                    alloc,
                    avg_roi,
                    stop_loss,
                    take_profit,
                    trailing_stop,
                    max_drawdown,
                    volatility_factor,
                )
                try:
                    price = price_lookup.get(token)
                    if price is None:
                        continue
                    await place_order_async(
                        token,
                        "buy",
                        amount,
                        price,
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                    if not dry_run:
                        await memory.log_trade(
                            token=token, direction="buy", amount=amount, price=price
                        )
                        await portfolio.update_async(
                            token,
                            amount,
                            price,
                        )
                except Exception as exc:  # pragma: no cover - trading errors
                    logging.warning("Buy failed for %s: %s", token, exc)
            elif should_sell(sims):
                logging.info("Selling %s", token)
                balance = portfolio.balances.get(token, 0.0)
                if balance <= 0:
                    continue
                try:
                    price = price_lookup.get(token)
                    if price is None:
                        continue
                    await place_order_async(
                        token,
                        "sell",
                        balance,
                        price,
                        testnet=testnet,
                        dry_run=dry_run,
                        keypair=keypair,
                    )
                    if not dry_run:
                        await memory.log_trade(
                            token=token, direction="sell", amount=balance, price=price
                        )
                        await portfolio.update_async(
                            token,
                            -balance,
                            price,
                        )
                except Exception as exc:  # pragma: no cover - trading errors
                    logging.warning("Sell failed for %s: %s", token, exc)
        return

    rm = RiskManager(portfolio, risk_metrics)
    va = recent_value_at_risk(memory, portfolio)
    rm.update_metrics(drawdown=drawdown, value_at_risk=va)

    for token in tokens:
        try:
            pred = await predict_price_movement(token)
        except Exception as exc:  # pragma: no cover - prediction errors
            logging.debug("Prediction failed for %s: %s", token, exc)
            continue
        decision = strategy_manager.decide(token, pred, price_lookup)
        if decision == "buy":
            amount = dynamic_order_size(
                portfolio.total_value(price_lookup),
                portfolio.percent_allocated(token, price_lookup),
                pred.expected_roi,
                stop_loss,
                take_profit,
                trailing_stop,
                max_drawdown,
                volatility_factor,
            )
            try:
                price = price_lookup.get(token)
                if price is None:
                    continue
                await place_order_async(
                    token,
                    "buy",
                    amount,
                    price,
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )
                if not dry_run:
                    await memory.log_trade(
                        token=token, direction="buy", amount=amount, price=price
                    )
                    await portfolio.update_async(
                        token,
                        amount,
                        price,
                    )
            except Exception as exc:  # pragma: no cover - trading errors
                logging.warning("Buy failed for %s: %s", token, exc)
        elif decision == "sell":
            balance = portfolio.balances.get(token, 0.0)
            if balance <= 0:
                continue
            try:
                price = price_lookup.get(token)
                if price is None:
                    continue
                await place_order_async(
                    token,
                    "sell",
                    balance,
                    price,
                    testnet=testnet,
                    dry_run=dry_run,
                    keypair=keypair,
                )
                if not dry_run:
                    await memory.log_trade(
                        token=token, direction="sell", amount=balance, price=price
                    )
                    await portfolio.update_async(
                        token,
                        -balance,
                        price,
                    )
            except Exception as exc:  # pragma: no cover - trading errors
                logging.warning("Sell failed for %s: %s", token, exc)

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
    data_val = cfg.get("rl_db_path", "offline_data.db")
    model_val = cfg.get("rl_model_path", "ppo_model.pt")
    data_path = Path(data_val)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    model_path = Path(model_val)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    algo = cfg.get("rl_algo", "ppo")
    policy = cfg.get("rl_policy", "mlp")
    auto_train = auto_train_cfg
    tune_interval = float(cfg.get("rl_tune_interval", rl_interval))
    cpu_count = detect_cpu_count()
    dyn_workers = bool(cfg.get("rl_dynamic_workers", cpu_count > 1))

    daemon = RLDaemon(
        memory_path=mem_db,
        data_path=str(data_path),
        model_path=str(model_path),
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


async def trading_loop(
    cfg: dict,
    runtime_cfg,
    memory,
    portfolio,
    state: TradingState,
    *,
    loop_delay: int,
    min_delay: int,
    max_delay: int,
    cpu_low_threshold: float,
    cpu_high_threshold: float,
    depth_freq_low: float,
    depth_freq_high: float,
    depth_rate_limit: float,
    iterations: int | None,
    testnet: bool,
    dry_run: bool,
    offline: bool,
    token_file: str | None,
    discovery_method: str,
    keypair,
    stop_loss: float | None,
    take_profit: float | None,
    trailing_stop: float | None,
    max_drawdown: float,
    volatility_factor: float,
    arbitrage_threshold: float,
    arbitrage_amount: float,
    strategy_manager,
    agent_manager,
    market_ws_url: str | None,
    order_book_ws_url: str | None,
    arbitrage_tokens: list[str] | None,
    rl_daemon: bool,
    rl_interval: float,
    proc_ref: list,
) -> None:
    """Main trading loop."""

    from . import event_bus, arbitrage, depth_client, order_book_ws
    from .market_ws import listen_and_trade
    from .data_pipeline import start_depth_snapshot_listener
    from .onchain_metrics import fetch_dex_metrics_async

    depth_updates = 0
    prev_count = 0
    prev_ts = time.monotonic()

    def _count(_p):
        nonlocal depth_updates
        depth_updates += 1

    unsub_counter = event_bus.subscribe("depth_update", _count)

    def _record_trade(payload):
        state.last_trade_times[payload.token] = datetime.datetime.utcnow()

    unsub_trade = event_bus.subscribe("trade_logged", _record_trade)
    bus_started = False
    bus_url = event_bus.get_event_bus_url()  # type: ignore[attr-defined]
    default_url = event_bus.DEFAULT_WS_URL  # type: ignore[attr-defined]
    check_url = bus_url or default_url
    try:
        ws = await event_bus.connect_ws(check_url)
        try:
            await ws.ping()
        finally:
            await event_bus.disconnect_ws()
    except Exception:
        await event_bus.start_ws_server()
        bus_started = True

    rl_task = await _init_rl_training(
        cfg, rl_daemon=rl_daemon, rl_interval=rl_interval
    )
    cfg_val = cfg.get("collect_offline_data")
    if cfg_val is not None:
        collect_data = str(cfg_val).strip().lower() in {"1", "true", "yes"}
    else:
        collect_data = parse_bool_env("COLLECT_OFFLINE_DATA", False)
    stop_collector = None
    if collect_data:
        db_val = cfg.get("rl_db_path", "offline_data.db")
        db_path = Path(db_val)
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        stop_collector = start_depth_snapshot_listener(str(db_path))
    prev_activity = 0.0
    iteration_idx = 0
    startup_reported = False

    timeout = float(os.getenv("FIRST_TRADE_TIMEOUT", "0") or 0)
    retry_first_trade = parse_bool_env("FIRST_TRADE_RETRY", False)
    first_trade_task = None

    def _check_timeout() -> None:
        if first_trade_task and first_trade_task.done():
            first_trade_task.result()

    def adjust_delay(metrics: dict) -> None:
        nonlocal loop_delay, prev_activity, prev_count, prev_ts, depth_rate_limit
        import psutil  # local import to avoid optional dependency issues

        activity = metrics.get("liquidity", 0.0) + metrics.get("volume", 0.0)
        cpu = psutil.cpu_percent() if psutil is not None else 0.0  # type: ignore
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

    use_depth_stream = parse_bool_env("USE_DEPTH_STREAM", True)

    async with asyncio.TaskGroup() as tg:
        if proc_ref[0]:
            tg.create_task(depth_service_watchdog(cfg, proc_ref))

        if timeout > 0:
            first_trade_task = tg.create_task(
                _check_first_trade(timeout, retry_first_trade)
            )

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

            tg.create_task(run_market_ws())

        if use_depth_stream:

            async def run_depth_ws() -> None:
                while True:
                    try:
                        await depth_client.listen_depth_ws()
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Depth websocket failed: %s", exc)
                        await asyncio.sleep(1.0)

            tg.create_task(run_depth_ws())

        if order_book_ws_url:

            async def run_order_book() -> None:
                while True:
                    try:
                        async for _ in order_book_ws.stream_order_book(
                            order_book_ws_url, rate_limit=depth_rate_limit
                        ):
                            pass
                    except Exception as exc:  # pragma: no cover - network errors
                        logging.error("Order book stream failed: %s", exc)
                        await asyncio.sleep(1.0)

            tg.create_task(run_order_book())

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

            tg.create_task(monitor_arbitrage())

    if iterations is None:
        while True:
            _check_timeout()
            if not startup_reported:
                metrics_aggregator.emit_startup_complete(
                    (time.perf_counter() - _PROCESS_START_TIME) * 1000.0
                )
                startup_reported = True
            await run_iteration(
                memory,
                portfolio,
                state,
                runtime_cfg,
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
            if state.last_tokens:
                metrics = await fetch_dex_metrics_async(
                    state.last_tokens[0], os.getenv("METRICS_BASE_URL")
                )
                adjust_delay(metrics)
            iteration_idx += 1
            if (
                agent_manager
                and iteration_idx % getattr(agent_manager, "evolve_interval", 1) == 0
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
            _check_timeout()
            if not startup_reported:
                metrics_aggregator.emit_startup_complete(
                    (time.perf_counter() - _PROCESS_START_TIME) * 1000.0
                )
                startup_reported = True
            await run_iteration(
                memory,
                portfolio,
                state,
                runtime_cfg,
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
            if state.last_tokens:
                metrics = await fetch_dex_metrics_async(
                    state.last_tokens[0], os.getenv("METRICS_BASE_URL")
                )
                adjust_delay(metrics)
            iteration_idx += 1
            if (
                agent_manager
                and iteration_idx % getattr(agent_manager, "evolve_interval", 1) == 0
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

    if rl_task:
        rl_task.cancel()
        with contextlib.suppress(Exception):
            await rl_task
    if stop_collector:
        stop_collector()
    if bus_started:
        await event_bus.stop_ws_server()
    unsub_trade()
    unsub_counter()

