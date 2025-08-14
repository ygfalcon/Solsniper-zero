import threading
import os
import asyncio
import json
from dataclasses import asdict, is_dataclass
import logging
import socket
import errno
from collections import deque
from typing import Any
import time
import subprocess
import sys
from pathlib import Path
from contextlib import nullcontext
import urllib.parse

from flask import Flask, Blueprint, jsonify, request, render_template_string

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .http import close_session
from .util import install_uvloop
from .event_bus import subscription, publish
try:
    import websockets
except Exception:  # pragma: no cover - optional
    websockets = None

import sqlalchemy as sa
import numpy as np
import sqlparse

from .config import (
    load_config,
    apply_env_overrides,
    set_env_from_config,
    initialize_event_bus,
    get_event_bus_url,
    get_depth_ws_addr,
    list_configs,
    save_config,
    select_config,
    get_active_config_name,
    load_selected_config,
)
from . import config as config_module
from .prices import fetch_token_prices
from .strategy_manager import StrategyManager
from . import wallet
from . import main as main_module
from .memory import Memory
from .base_memory import BaseMemory
from .portfolio import Portfolio

# shared in-memory database instance used by UI routes
MEMORY: BaseMemory = Memory("sqlite:///memory.db")


def set_memory(memory: BaseMemory) -> None:
    """Override the global memory instance (mainly for tests)."""
    global MEMORY
    MEMORY = memory

logger = logging.getLogger(__name__)

# websocket ping configuration
_WS_PING_INTERVAL = float(os.getenv("WS_PING_INTERVAL", "20") or 20)
_WS_PING_TIMEOUT = float(os.getenv("WS_PING_TIMEOUT", "20") or 20)

# event websocket configuration
_EVENT_WS_PORT = int(os.getenv("EVENT_WS_PORT", "8770") or 8770)

bp = Blueprint("ui", __name__)

# The test suite provides a very small stub of :mod:`flask` that lacks the
# ``record_once`` helper normally available on real ``Blueprint`` instances.
# When running with this stub, accessing ``bp.record_once`` would raise an
# ``AttributeError`` at import time.  To keep the module importable in the
# minimal test environment we provide a no-op fallback that simply returns the
# decorated function unchanged.
if not hasattr(bp, "record_once"):
    def _record_once(func):  # pragma: no cover - simple fallback
        return func

    bp.record_once = _record_once  # type: ignore[attr-defined]

_DEFAULT_PRESET = Path(__file__).resolve().parent.parent / "config" / "default.toml"

# in-memory log storage for UI access (initialised in ``create_app``)
log_buffer: deque[str] = deque()
buffer_handler: logging.Handler | None = None
_SUBSCRIPTIONS: list[Any] = []


def _check_redis_connection() -> None:
    """Warn the user when Redis is unreachable."""
    url = get_event_bus_url()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"redis", "rediss"}:
        return
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=0.1):
            return
    except OSError:
        try:
            subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(10):
                time.sleep(0.5)
                try:
                    with socket.create_connection((host, port), timeout=0.1):
                        return
                except OSError:
                    continue
        except Exception:
            pass
        logger.error(
            "Failed to connect to Redis at %s:%s. Start redis-server or set EVENT_BUS_URL.",
            host,
            port,
        )

# Ensure event bus subscriptions are cleaned up when the application context ends
def _clear_subscriptions(_exc: Exception | None) -> None:
    """Terminate all active event bus subscriptions."""
    for sub in _SUBSCRIPTIONS:
        try:
            sub.__exit__(None, None, None)
        except Exception:
            pass
    _SUBSCRIPTIONS.clear()


@bp.record_once
def _register_clear_subscriptions(state: Any) -> None:
    """Register teardown to clear event bus subscriptions once app is ready."""
    state.app.teardown_appcontext(_clear_subscriptions)

# websocket state for streaming log lines
log_ws_clients: set[Any] = set()
log_ws_loop: asyncio.AbstractEventLoop | None = None


def _broadcast_log_line(line: str) -> None:
    """Broadcast a log line to all connected websocket clients."""
    if log_ws_loop is None:
        return

    async def _broadcast() -> None:
        to_remove: list[Any] = []
        for ws in list(log_ws_clients):
            try:
                await ws.send(line)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            log_ws_clients.discard(ws)

    asyncio.run_coroutine_threadsafe(_broadcast(), log_ws_loop)


class _BufferHandler(logging.Handler):
    """Logging handler that stores formatted log records in ``log_buffer``."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        line = self.format(record)
        log_buffer.append(line)
        _broadcast_log_line(line)

def _update_weights(weights):
    if is_dataclass(weights):
        weights = asdict(weights)["weights"]
    try:
        os.environ["AGENT_WEIGHTS"] = json.dumps(weights)
    except Exception:
        pass

def _update_rl_weights(msg: Any) -> None:
    weights = msg.weights if hasattr(msg, "weights") else msg.get("weights", {})
    risk = msg.risk if hasattr(msg, "risk") else msg.get("risk", {})
    if isinstance(weights, dict):
        try:
            os.environ["AGENT_WEIGHTS"] = json.dumps(weights)
        except Exception:
            pass
    rm = risk.get("risk_multiplier") if isinstance(risk, dict) else None
    if rm is not None:
        os.environ["RISK_MULTIPLIER"] = str(rm)



def _store_rl_metrics(msg: Any) -> None:
    loss = getattr(msg, "loss", None)
    if loss is None and isinstance(msg, dict):
        loss = msg.get("loss")
    reward = getattr(msg, "reward", None)
    if reward is None and isinstance(msg, dict):
        reward = msg.get("reward")
    if loss is None or reward is None:
        return
    rl_metrics.append({"loss": float(loss), "reward": float(reward)})



def _store_system_metrics(msg: Any) -> None:
    cpu = getattr(msg, "cpu", None)
    mem = getattr(msg, "memory", None)
    if cpu is None and isinstance(msg, dict):
        cpu = msg.get("cpu")
    if mem is None and isinstance(msg, dict):
        mem = msg.get("memory")
    if cpu is None or mem is None:
        return
    system_metrics["cpu"] = float(cpu)
    system_metrics["memory"] = float(mem)



async def _send_rl_update(payload):
    if rl_ws_loop is None:
        return
    if is_dataclass(payload):
        payload = asdict(payload)
    msg = json.dumps(payload)

    async def _broadcast():
        to_remove = []
        for ws in list(rl_ws_clients):
            try:
                await ws.send(msg)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            rl_ws_clients.discard(ws)

    asyncio.run_coroutine_threadsafe(_broadcast(), rl_ws_loop)




def _emit_ws_event(topic: str, payload: Any) -> None:
    """Send event payload to all connected websocket clients."""
    if event_ws_loop is None:
        return
    if is_dataclass(payload):
        payload = asdict(payload)
    msg = json.dumps({"topic": topic, "payload": payload})

    async def _broadcast() -> None:
        to_remove: list[Any] = []
        for ws in list(event_ws_clients):
            try:
                await ws.send(msg)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            event_ws_clients.discard(ws)

    asyncio.run_coroutine_threadsafe(_broadcast(), event_ws_loop)


def _sub_handler(topic: str):
    def handler(payload: Any) -> None:
        _emit_ws_event(topic, payload)

    return handler



depth_service_connected = False
last_heartbeat = 0.0
rl_daemon_heartbeat = 0.0
depth_service_heartbeat = 0.0

def _heartbeat(_payload: Any) -> None:
    global last_heartbeat, rl_daemon_heartbeat, depth_service_heartbeat
    service = getattr(_payload, "service", None)
    last_heartbeat = time.time()
    if service == "rl_daemon":
        rl_daemon_heartbeat = last_heartbeat
    elif service == "depth_service":
        depth_service_heartbeat = last_heartbeat


def _depth_status(payload: Any) -> None:
    global depth_service_connected
    status = str(getattr(payload, "status", payload.get("status")))
    depth_service_connected = status in {"connected", "reconnected"}


trading_thread = None
stop_event = threading.Event()
loop_delay = 60

# background thread/process for running scripts/start_all.py
start_all_thread = None
start_all_proc = None
start_all_ready = threading.Event()
startup_message = ""

# currently active portfolio and keypair used by the trading loop
current_portfolio: Portfolio | None = None
current_keypair = None
state_lock = threading.RLock()
pnl_history: list[float] = []
token_pnl_history: dict[str, list[float]] = {}
allocation_history: dict[str, list[float]] = {}
_HISTORY_LIMIT = 100
last_prices: dict[str, float] = {}

# most recent RL training metrics
rl_metrics: list[dict[str, float]] = []

# latest system metrics data
system_metrics: dict[str, float] = {"cpu": 0.0, "memory": 0.0}

# global RL training daemon for status reporting
rl_daemon = None

# store clients connected to RL checkpoint websocket
rl_ws_clients: set[Any] = set()
rl_ws_loop: asyncio.AbstractEventLoop | None = None

# store clients connected to event broadcast websocket
event_ws_clients: set[Any] = set()
event_ws_loop: asyncio.AbstractEventLoop | None = None

# ``BIRDEYE_API_KEY`` is optional when ``SOLANA_RPC_URL`` is provided for
# on-chain scanning.
REQUIRED_ENV_VARS = ("DEX_BASE_URL",)


def ensure_active_keypair() -> None:
    """Select the sole available keypair if none is active.

    When a single keypair exists in the keypair directory and no keypair
    is currently active, this helper selects it and populates the
    ``KEYPAIR_PATH`` environment variable if it is unset.  Wallet
    interaction errors are propagated to the caller.
    """

    if wallet.get_active_keypair_name() is not None:
        return
    keys = wallet.list_keypairs()
    if len(keys) != 1:
        return
    wallet.select_keypair(keys[0])
    if not os.getenv("KEYPAIR_PATH"):
        os.environ["KEYPAIR_PATH"] = os.path.join(
            wallet.KEYPAIR_DIR, keys[0] + ".json"
        )


def ensure_active_config() -> None:
    """Select the sole available config if none is active."""

    if get_active_config_name() is not None:
        return
    configs = list_configs()
    if len(configs) != 1:
        return
    select_config(configs[0])
    set_env_from_config(load_selected_config())
    _check_redis_connection()
    initialize_event_bus()


def record_history(prices: dict[str, float]) -> None:
    """Record PnL and allocation history using ``prices``."""
    last_prices.update(prices)
    with state_lock:
        pf = current_portfolio or Portfolio()
        entry = sum(p.amount * p.entry_price for p in pf.balances.values())
        value = sum(
            p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items()
        )
        pnl = value - entry
        pnl_history.append(pnl)
        if len(pnl_history) > _HISTORY_LIMIT:
            del pnl_history[:-_HISTORY_LIMIT]
        total = value
        for token, pos in pf.balances.items():
            price = prices.get(token, pos.entry_price)
            token_pnl_history.setdefault(token, []).append(
                (price - pos.entry_price) * pos.amount
            )
            if len(token_pnl_history[token]) > _HISTORY_LIMIT:
                del token_pnl_history[token][:-_HISTORY_LIMIT]
            alloc = (pos.amount * price) / total if total else 0.0
            allocation_history.setdefault(token, []).append(alloc)
            if len(allocation_history[token]) > _HISTORY_LIMIT:
                del allocation_history[token][:-_HISTORY_LIMIT]


def _on_price_update(msg: Any) -> None:
    token = getattr(msg, "token", None)
    price = getattr(msg, "price", None)
    if token is None and isinstance(msg, dict):
        token = msg.get("token")
        price = msg.get("price")
    if not isinstance(token, str) or not isinstance(price, (int, float)):
        return
    last_prices[token] = float(price)
    with state_lock:
        pf = current_portfolio or Portfolio()
        prices = {tok: last_prices.get(tok, pos.entry_price) for tok, pos in pf.balances.items()}
    if prices:
        record_history(prices)


def _on_portfolio_update(_msg: Any) -> None:
    with state_lock:
        pf = current_portfolio or Portfolio()
        tokens = list(pf.balances.keys())
    if not tokens:
        record_history({})
        return
    prices = fetch_token_prices(tokens)
    record_history(prices)


def _missing_required() -> list[str]:
    """Return names of required variables that are unset."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if not (os.getenv("BIRDEYE_API_KEY") or os.getenv("SOLANA_RPC_URL")):
        missing.append("BIRDEYE_API_KEY or SOLANA_RPC_URL")
    return missing


def create_app() -> Flask:
    """Create and configure the Flask application."""
    global log_buffer, buffer_handler, _SUBSCRIPTIONS

    install_uvloop()

    cfg = load_config()
    if not cfg and _DEFAULT_PRESET.is_file():
        cfg = load_config(_DEFAULT_PRESET)

    if get_active_config_name() is None and _DEFAULT_PRESET.is_file():
        dest = Path(config_module.CONFIG_DIR) / _DEFAULT_PRESET.name
        if not dest.exists():
            dest.write_bytes(_DEFAULT_PRESET.read_bytes())
        select_config(dest.name)

    cfg = apply_env_overrides(cfg)
    set_env_from_config(cfg)
    _check_redis_connection()
    initialize_event_bus()

    try:
        ensure_active_keypair()
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )

    ensure_active_config()

    # assemble startup summary
    active_cfg = get_active_config_name() or "<none>"
    active_keypair = wallet.get_active_keypair_name() or "<none>"
    solana_rpc = os.getenv("SOLANA_RPC_URL") or cfg.get("solana_rpc_url")
    jito_rpc = os.getenv("JITO_RPC_URL") or cfg.get("jito_rpc_url")
    event_bus_url = get_event_bus_url(cfg)
    depth_addr, depth_port = get_depth_ws_addr(cfg)
    jito_ws = os.getenv("JITO_WS_URL") or cfg.get("jito_ws_url")
    order_book_ws = os.getenv("ORDER_BOOK_WS_URL") or cfg.get("order_book_ws_url")

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Config", active_cfg)
    table.add_row("Keypair", active_keypair)
    if solana_rpc:
        table.add_row("Solana RPC", str(solana_rpc))
    if jito_rpc:
        table.add_row("Jito RPC", str(jito_rpc))
    if event_bus_url:
        table.add_row("Event Bus", str(event_bus_url))
    if jito_ws:
        table.add_row("Jito WS", str(jito_ws))
    if order_book_ws:
        table.add_row("OrderBook WS", str(order_book_ws))
    if depth_addr:
        table.add_row("Depth WS", f"{depth_addr}:{depth_port}")

    banner = Panel(
        table,
        title="[bold green]Solhunter Zero Startup[/bold green]",
        border_style="green",
    )
    Console().print(banner)

    app = Flask(
        __name__, static_folder=str(Path(__file__).resolve().parent / "static")
    )
    app.register_blueprint(bp)

    log_buffer = deque(maxlen=200)
    buffer_handler = _BufferHandler()
    buffer_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(buffer_handler)

    _SUBSCRIPTIONS = [
        subscription("weights_updated", _update_weights),
        subscription("rl_weights", _update_rl_weights),
        subscription("rl_metrics", _store_rl_metrics),
        subscription("system_metrics_combined", _store_system_metrics),
        subscription("rl_checkpoint", _send_rl_update),
        subscription("rl_weights", _send_rl_update),
        subscription("rl_metrics", _send_rl_update),
        subscription("action_executed", _sub_handler("action_executed")),
        subscription("weights_updated", _sub_handler("weights_updated")),
        subscription("rl_weights", _sub_handler("rl_weights")),
        subscription("rl_metrics", _sub_handler("rl_metrics")),
        subscription("risk_updated", _sub_handler("risk_updated")),
        subscription("config_updated", _sub_handler("config_updated")),
        subscription("system_metrics_combined", _sub_handler("system_metrics")),
        subscription("heartbeat", _heartbeat),
        subscription("depth_service_status", _depth_status),
        subscription("price_update", _on_price_update),
        subscription("portfolio_updated", _on_portfolio_update),
    ]
    for sub in _SUBSCRIPTIONS:
        sub.__enter__()
    global startup_message
    active_keypair = wallet.get_active_keypair_name()
    active_config = get_active_config_name()
    if active_keypair and active_config:
        startup_message = (
            f"Detected keypair '{active_keypair}' and config '{active_config}'. "
            "Starting trading."
        )
        logging.getLogger(__name__).info(startup_message)
        try:
            with app.app_context():
                start()
        except Exception as exc:
            logging.getLogger(__name__).error("Automatic start failed: %s", exc)
    else:
        missing: list[str] = []
        if not active_keypair:
            missing.append("keypair")
        if not active_config:
            missing.append("configuration")
        startup_message = (
            "Missing required "
            + " and ".join(missing)
            + ". Configure them to begin trading."
        )
        logging.getLogger(__name__).warning(startup_message)

    if os.getenv("AUTO_START") == "1":
        ctx = app.app_context() if hasattr(app, "app_context") else nullcontext()
        try:
            with ctx:
                result = autostart()
            data = result.get_json() if hasattr(result, "get_json") else result
            logging.warning("Autostart triggered: %s", data.get("status"))
        except Exception as exc:  # pragma: no cover - simple logging
            logging.warning("Autostart failed: %s", exc)

    return app
async def trading_loop(memory: BaseMemory | None = None) -> None:
    global current_portfolio, current_keypair

    cfg = apply_env_overrides(load_selected_config())
    set_env_from_config(cfg)
    _check_redis_connection()
    initialize_event_bus()
    ensure_active_config()

    try:
        ensure_active_keypair()
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    memory = memory or MEMORY
    portfolio = Portfolio()
    state = main_module.TradingState()

    with state_lock:
        current_portfolio = portfolio
    keypair_path = os.getenv("KEYPAIR_PATH")
    try:
        env_keypair = (
            await wallet.load_keypair_async(keypair_path) if keypair_path else None
        )
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with state_lock:
        current_keypair = env_keypair

    while not stop_event.is_set():
        try:
            selected_keypair = await wallet.load_selected_keypair_async()
        except Exception as exc:
            print(
                f"Wallet interaction failed: {exc}\n"
                "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        keypair = selected_keypair if selected_keypair is not None else env_keypair
        with state_lock:
            current_keypair = keypair
        await main_module._run_iteration(
            memory,
            portfolio,
            state,
            testnet=False,
            dry_run=False,
            offline=False,
            keypair=keypair,
        )
        for _ in range(loop_delay):
            if stop_event.is_set():
                break
            await asyncio.sleep(1)


@bp.route("/start", methods=["POST"])
def start() -> dict:
    global trading_thread
    if trading_thread and trading_thread.is_alive():
        return jsonify({"status": "already running"})

    cfg = apply_env_overrides(load_config("config.toml"))
    set_env_from_config(cfg)
    _check_redis_connection()
    initialize_event_bus()
    ensure_active_config()

    try:
        from . import data_sync

        def _background_sync() -> None:
            try:
                asyncio.run(data_sync.sync_recent())
            except Exception as exc:  # pragma: no cover - ignore sync errors
                logger.warning("data sync failed: %s", exc)

        threading.Thread(target=_background_sync, daemon=True).start()
    except Exception as exc:  # pragma: no cover - ignore sync errors
        logger.warning("data sync failed: %s", exc)

    try:
        ensure_active_keypair()
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )
        return jsonify(
            {
                "status": "error",
                "message": "wallet unavailable; run solhunter-wallet or set MNEMONIC",
            }
        ), 500

    missing = _missing_required()
    if missing:
        msg = "Missing required configuration: " + ", ".join(missing)
        return jsonify({"status": "error", "message": msg}), 400

    stop_event.clear()
    trading_thread = threading.Thread(
        target=lambda: asyncio.run(trading_loop()), daemon=True
    )
    trading_thread.start()
    return jsonify({"status": "started"})


@bp.route("/stop", methods=["POST"])
def stop() -> dict:
    stop_event.set()
    if trading_thread:
        trading_thread.join()
    return jsonify({"status": "stopped"})


@bp.route("/autostart", methods=["POST"])
def autostart() -> dict:
    """Launch the bot in fully automatic mode."""
    global trading_thread
    if trading_thread and trading_thread.is_alive():
        return jsonify({"status": "already running"})
    trading_thread = threading.Thread(
        target=main_module.run_auto, daemon=True
    )
    trading_thread.start()
    return jsonify({"status": "started"})


def _run_start_all() -> None:
    """Run scripts/start_all.py in a subprocess and wait for it to exit."""
    global start_all_proc
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "start_all.py"),
        "autopilot",
    ]
    start_all_proc = subprocess.Popen(cmd)
    start_all_ready.set()
    start_all_proc.wait()


@bp.route("/start_all", methods=["POST"])
def start_all_route() -> dict:
    global start_all_thread
    if start_all_thread and start_all_thread.is_alive():
        return jsonify({"status": "already running"})
    start_all_ready.clear()
    start_all_thread = threading.Thread(target=_run_start_all, daemon=True)
    start_all_thread.start()
    start_all_ready.wait(0.2)
    return jsonify({"status": "started"})


@bp.route("/stop_all", methods=["POST"])
def stop_all_route() -> dict:
    if start_all_proc and start_all_proc.poll() is None:
        start_all_proc.terminate()
        try:
            start_all_proc.wait(timeout=5)
        except Exception:
            start_all_proc.kill()
    if start_all_thread:
        start_all_thread.join(timeout=5)
    return jsonify({"status": "stopped"})


@bp.route("/risk", methods=["GET", "POST"])
def risk_params() -> dict:
    method = getattr(request, "method", None)
    if method is None:
        method = "POST" if getattr(request, "json", None) is not None else "GET"
    if method == "POST":
        if hasattr(request, "get_json"):
            data = request.get_json() or {}
        else:
            data = getattr(request, "json", None) or {}
        rt = data.get("risk_tolerance")
        ma = data.get("max_allocation")
        rm = data.get("risk_multiplier")
        updates: dict[str, float] = {}
        if rt is not None:
            try:
                rt = float(rt)
            except ValueError:
                return jsonify({"error": "invalid numeric value"}), 400
            os.environ["RISK_TOLERANCE"] = str(rt)
            updates["risk_tolerance"] = rt
        if ma is not None:
            try:
                ma = float(ma)
            except ValueError:
                return jsonify({"error": "invalid numeric value"}), 400
            os.environ["MAX_ALLOCATION"] = str(ma)
            updates["max_allocation"] = ma
        if rm is not None:
            try:
                rm = float(rm)
            except ValueError:
                return jsonify({"error": "invalid numeric value"}), 400
            os.environ["RISK_MULTIPLIER"] = str(rm)
            updates["risk_multiplier"] = rm
        if updates:
            publish("risk_updated", updates)
        return jsonify({"status": "ok"})
    rt_raw = os.getenv("RISK_TOLERANCE", "0.1")
    try:
        rt = float(rt_raw)
    except ValueError:
        logger.warning("Invalid RISK_TOLERANCE '%s'; using default 0.1", rt_raw)
        rt = 0.1

    ma_raw = os.getenv("MAX_ALLOCATION", "0.2")
    try:
        ma = float(ma_raw)
    except ValueError:
        logger.warning("Invalid MAX_ALLOCATION '%s'; using default 0.2", ma_raw)
        ma = 0.2

    rm_raw = os.getenv("RISK_MULTIPLIER", "1.0")
    try:
        rm = float(rm_raw)
    except ValueError:
        logger.warning("Invalid RISK_MULTIPLIER '%s'; using default 1.0", rm_raw)
        rm = 1.0

    return jsonify(
        {
            "risk_tolerance": rt,
            "max_allocation": ma,
            "risk_multiplier": rm,
        }
    )


@bp.route("/weights", methods=["GET", "POST"])
def agent_weights() -> dict:
    """Get or update agent weighting factors."""
    method = getattr(request, "method", None)
    if method is None:
        method = "POST" if getattr(request, "json", None) is not None else "GET"
    if method == "POST":
        getter = getattr(request, "get_json", None)
        weights = getter() if callable(getter) else getattr(request, "json", {})
        weights = weights or {}
        os.environ["AGENT_WEIGHTS"] = json.dumps(weights)
        return jsonify({"status": "ok"})

    env = os.getenv("AGENT_WEIGHTS")
    if not env:
        return jsonify({})
    try:
        return jsonify(json.loads(env))
    except Exception:
        try:
            import ast

            return jsonify(ast.literal_eval(env))
        except Exception:
            return jsonify({})


@bp.route("/strategies", methods=["GET", "POST"])
def strategies_route() -> dict:
    """Get or set active strategy modules."""
    if request.method == "POST":
        names = request.get_json() or []
        if isinstance(names, list):
            os.environ["STRATEGIES"] = ",".join(names)
        else:
            os.environ.pop("STRATEGIES", None)
        return jsonify({"status": "ok"})

    env = os.getenv("STRATEGIES")
    active = [s.strip() for s in env.split(",") if s.strip()] if env else []
    if not active:
        active = list(StrategyManager.DEFAULT_STRATEGIES)
    return jsonify({"available": list(StrategyManager.DEFAULT_STRATEGIES), "active": active})


@bp.route("/discovery", methods=["GET", "POST"])
def discovery_method() -> dict:
    method = getattr(request, "method", None)
    if method is None:
        method = "POST" if getattr(request, "json", None) is not None else "GET"
    if method == "POST":
        getter = getattr(request, "get_json", None)
        data = getter() if callable(getter) else getattr(request, "json", {})
        method = data.get("method") if isinstance(data, dict) else None
        allowed = {"websocket", "mempool"}
        if method:
            if method not in allowed:
                msg = (
                    f"Invalid discovery method '{method}'. "
                    f"Allowed methods: {', '.join(sorted(allowed))}"
                )
                return jsonify({"error": msg}), 400
            os.environ["DISCOVERY_METHOD"] = method
        return jsonify({"status": "ok"})
    return jsonify({"method": os.getenv("DISCOVERY_METHOD", "websocket")})


@bp.route("/keypairs", methods=["GET"])
def keypairs() -> dict:
    try:
        data = {
            "keypairs": wallet.list_keypairs(),
            "active": wallet.get_active_keypair_name(),
        }
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )
        return jsonify({"error": "wallet unavailable"}), 500
    return jsonify(data)


@bp.route("/keypairs/upload", methods=["POST"])
def upload_keypair() -> dict:
    file = request.files.get("file")
    raw_name = request.form.get("name") or (file.filename if file else None)
    if not file or not raw_name:
        return jsonify({"error": "missing file or name"}), 400
    name = os.path.basename(raw_name)
    if (
        name != raw_name
        or ".." in name
        or any(sep in name for sep in ("/", "\\"))
    ):
        return jsonify({"error": "invalid name"}), 400
    data = file.read()
    try:
        wallet.save_keypair(name, list(json.loads(data)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok"})


@bp.route("/keypairs/select", methods=["POST"])
def select_keypair_route() -> dict:
    name = request.get_json().get("name")
    try:
        wallet.select_keypair(name)
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "wallet unavailable; run solhunter-wallet or set MNEMONIC",
                }
            ),
            400,
        )
    return jsonify({"status": "ok"})


@bp.route("/configs", methods=["GET"])
def configs() -> dict:
    return jsonify({"configs": list_configs(), "active": get_active_config_name()})


@bp.route("/configs/upload", methods=["POST"])
def upload_config() -> dict:
    file = request.files.get("file")
    raw_name = request.form.get("name") or (file.filename if file else None)
    if not file or not raw_name:
        return jsonify({"error": "missing file or name"}), 400
    name = os.path.basename(raw_name)
    if (
        name != raw_name
        or ".." in name
        or any(sep in name for sep in ("/", "\\"))
    ):
        return jsonify({"error": "invalid name"}), 400
    try:
        save_config(name, file.read())
    except Exception as exc:  # mirror keypair upload behavior
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok"})


@bp.route("/configs/select", methods=["POST"])
def select_config_route() -> dict:
    name = request.get_json().get("name")
    select_config(name)
    return jsonify({"status": "ok"})


@bp.route("/positions")
def positions() -> dict:
    with state_lock:
        pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    result = {}
    for token, pos in pf.balances.items():
        price = prices.get(token, pos.entry_price)
        roi = pf.position_roi(token, price)
        result[token] = {
            "amount": pos.amount,
            "entry_price": pos.entry_price,
            "current_price": price,
            "roi": roi,
        }
    return jsonify(result)


@bp.route("/trades")
def trades() -> dict:
    mem = MEMORY
    recents = [
        {
            "token": t.token,
            "direction": t.direction,
            "amount": t.amount,
            "price": t.price,
            "timestamp": t.timestamp.isoformat(),
        }
        for t in mem.list_trades(limit=50)
    ]
    return jsonify(recents)


@bp.route("/vars")
def vars_route() -> dict:
    """Return recent VaR measurements."""
    mem = MEMORY
    data = [
        {"value": v.value, "timestamp": v.timestamp.isoformat()}
        for v in mem.list_vars()[-50:]
    ]
    return jsonify(data)


@bp.route("/roi")
def roi() -> dict:
    with state_lock:
        pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    entry = sum(p.amount * p.entry_price for p in pf.balances.values())
    value = sum(
        p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items()
    )
    roi = (value - entry) / entry if entry else 0.0
    return jsonify({"roi": roi})


@bp.route("/exposure")
def exposure() -> dict:
    """Return current portfolio weights."""
    with state_lock:
        pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    weights = pf.weights(prices)
    return jsonify(weights)


@bp.route("/sharpe")
def sharpe_ratio() -> dict:
    """Return rolling Sharpe ratio from PnL history."""
    if len(pnl_history) < 2:
        return jsonify({"sharpe": 0.0})
    arr = np.array(pnl_history, dtype=float)
    returns = np.diff(arr)
    if returns.size == 0:
        return jsonify({"sharpe": 0.0})
    mean = float(returns.mean())
    std = float(returns.std())
    sharpe = mean / std if std > 0 else 0.0
    return jsonify({"sharpe": sharpe})


@bp.route("/pnl")
def pnl() -> dict:
    with state_lock:
        pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    entry = sum(p.amount * p.entry_price for p in pf.balances.values())
    value = sum(
        p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items()
    )
    pnl = value - entry
    return jsonify({"pnl": pnl, "history": pnl_history})


@bp.route("/token_history")
def token_history() -> dict:
    with state_lock:
        pf = current_portfolio or Portfolio()
    result: dict[str, dict[str, list[float]]] = {}
    for token in pf.balances:
        result[token] = {
            "pnl_history": token_pnl_history.get(token, []),
            "allocation_history": allocation_history.get(token, []),
        }
    return jsonify(result)


@bp.route("/balances")
def balances() -> dict:
    """Return portfolio balances with USD values."""
    with state_lock:
        pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    result = {}
    for token, pos in pf.balances.items():
        price = prices.get(token)
        usd = pos.amount * price if price is not None else None
        result[token] = {"amount": pos.amount, "price": price, "usd": usd}
    return jsonify(result)


@bp.route("/logs")
def logs() -> dict:
    """Return recent log messages."""
    return jsonify({"logs": list(log_buffer)})


@bp.route("/rl/status")
def rl_status() -> dict:
    """Return RL training metrics if available."""
    if rl_daemon is None:
        return jsonify({"last_train_time": None, "checkpoint_path": None, "metrics": rl_metrics})
    return jsonify(
        {
            "last_train_time": getattr(rl_daemon, "last_train_time", None),
            "checkpoint_path": getattr(rl_daemon, "checkpoint_path", None),
            "metrics": rl_metrics,
        }
    )


@bp.route("/status")
def status() -> dict:
    """Return status of background components."""
    trading_alive = trading_thread.is_alive() if trading_thread else False
    rl_alive = rl_daemon is not None or time.time() - rl_daemon_heartbeat < 120
    depth_alive = depth_service_connected or time.time() - depth_service_heartbeat < 120
    heartbeat_alive = time.time() - last_heartbeat < 120
    event_alive = False
    url = get_event_bus_url()
    if url and websockets is not None:
        async def _check():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                ):
                    return True
            except Exception:
                return False

        try:
            event_alive = asyncio.run(_check())
        except Exception:
            event_alive = False
    data = {
        "trading_loop": trading_alive,
        "rl_daemon": rl_alive,
        "depth_service": depth_alive,
        "event_bus": event_alive,
        "heartbeat": heartbeat_alive,
        "system_metrics": system_metrics,
    }
    if getattr(request, "args", None) and request.args.get("include_message"):
        data["message"] = startup_message
    return jsonify(data)


def _validate_sql(sql: str, allowed: set[str]) -> bool:
    """Return ``True`` if *sql* contains a single allowed statement.

    Comments are stripped before parsing and only one statement is permitted.
    The first keyword of the statement must match one of the entries in
    ``allowed``.  Any additional statements or unknown commands will result in
    ``False`` being returned.
    """

    try:
        stripped = sqlparse.format(sql, strip_comments=True).strip()
        statements = [s for s in sqlparse.parse(stripped) if s and not s.is_whitespace]
    except Exception:  # pragma: no cover - invalid SQL
        return False
    if len(statements) != 1:
        return False
    token = statements[0].token_first(skip_ws=True)
    if token is None:
        return False
    return token.value.lower() in allowed


def _authorized() -> bool:
    """Check simple token based auth for memory endpoints."""
    token = os.getenv("UI_API_TOKEN")
    if not token:
        return True
    headers = getattr(request, "headers", {}) or {}
    header = headers.get("Authorization", "") if hasattr(headers, "get") else ""
    return header == f"Bearer {token}"


def _get_request_json() -> dict:
    """Return JSON payload from the current request in both real and stub envs."""
    getter = getattr(request, "get_json", None)
    if callable(getter):
        return getter() or {}
    return getattr(request, "json", {}) or {}


@bp.route("/memory/insert", methods=["POST"])
def memory_insert() -> dict:
    data = _get_request_json()
    sql = data.get("sql")
    params = data.get("params", {})
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    if not _validate_sql(sql, {"insert"}):
        return jsonify({"error": "disallowed sql"}), 400
    mem = MEMORY
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        session.commit()
    return jsonify({"status": "ok", "rows": result.rowcount})


@bp.route("/memory/update", methods=["POST"])
def memory_update() -> dict:
    data = _get_request_json()
    sql = data.get("sql")
    params = data.get("params", {})
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    if not _validate_sql(sql, {"update", "delete"}):
        return jsonify({"error": "disallowed sql"}), 400
    mem = MEMORY
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        session.commit()
    return jsonify({"status": "ok", "rows": result.rowcount})


@bp.route("/memory/query", methods=["POST"])
def memory_query() -> dict:
    data = _get_request_json()
    sql = data.get("sql")
    params = data.get("params", {})
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    if not _validate_sql(sql, {"select"}):
        return jsonify({"error": "disallowed sql"}), 400
    mem = MEMORY
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        rows = [dict(row._mapping) for row in result]
    return jsonify(rows)


HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>SolHunter UI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
</head>
<body>
    <div class="container">
    <button id='start' class='action-btn'>
        <i class="fa-solid fa-play"></i> Start
    </button>
    <button id='stop' class='action-btn'>
        <i class="fa-solid fa-stop"></i> Stop
    </button>
    <a href="/startup" class="action-btn" target="_blank">View Logs</a>
    <select id='keypair_select'></select>
    <pre id='status_info'></pre>
    <pre id='logs'></pre>
    <p>Active Keypair: <span id='active_keypair'></span></p>
    <p>Active Config: <span id='active_config'></span></p>
    <div class="section">
        <h3>Strategies</h3>
        <div id='strategy_controls'></div>
        <button id='save_strategies'>Save Strategies</button>
    </div>

    <div class="section">
        <h3>ROI: <span id='roi_value'>0</span></h3>
        <canvas id='roi_chart' width='400' height='100'></canvas>
        <p id='roi_legend' title='Green indicates positive ROI, red indicates negative ROI.'>
            ROI color legend: green ≥ 0, red &lt; 0
        </p>
    </div>

    <div class="section">
        <h3>Positions</h3>
        <pre id='positions'></pre>
    </div>

    <div class="section">
        <h3>Recent Trades</h3>
        <pre id='trades'></pre>
        <canvas id='trade_chart' width='400' height='100'></canvas>
    </div>

    <div class="section">
        <h3>Agent Weights</h3>
        <div id='weights_controls'></div>
        <button id='save_weights'>Save Weights</button>
        <canvas id='weights_chart' width='400' height='100'></canvas>
    </div>

    <div class="section">
        <h3>Token PnL</h3>
        <canvas id='pnl_chart' width='400' height='100'></canvas>
    </div>

    <div class="section">
        <h3>Token Allocation</h3>
        <canvas id='allocation_chart' width='400' height='100'></canvas>
    </div>

    <div class="section">
        <h3>VaR History</h3>
        <pre id='var_values'></pre>
        <canvas id='var_chart' width='400' height='100'></canvas>
    </div>

    <div class="section">
        <h3>Exposure</h3>
        <pre id='exposure'></pre>
    </div>

    <div class="section">
        <h3>Sharpe Ratio: <span id='sharpe_val'>0</span></h3>

        <h3>RL Status</h3>
        <pre id='rl_status'></pre>
    </div>

    <div class="section">
        <h3>Logs</h3>
        <pre id='logs'></pre>
    </div>

    <div class="section">
        <h3>Risk Parameters</h3>
        <label>Risk Tolerance <input id='risk_tolerance' type='number' step='0.01'></label>
        <label>Max Allocation <input id='max_allocation' type='number' step='0.01'></label>
        <label>Risk Multiplier <input id='risk_multiplier' type='number' step='0.01'></label>
        <button id='save_risk'>Save Risk</button>
    </div>
    </div>

    <script>
    document.getElementById('start').onclick = function() {
        fetch('/start_all', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop_all', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    let logIndex = 0;

    const roiChart = new Chart(document.getElementById('roi_chart'), {
        type: 'line',
        data: {labels: [], datasets: [{label: 'ROI', data: [], borderColor: [], backgroundColor: []}]},
        options: {scales: {y: {beginAtZero: true}}}
    });

    const tradeChart = new Chart(document.getElementById('trade_chart'), {
        type: 'bar',
        data: {labels: ['buy', 'sell'], datasets: [{label:'Trades', data:[0,0]}]},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const weightsChart = new Chart(document.getElementById('weights_chart'), {
        type: 'bar',
        data: {labels: [], datasets: [{label:'Weight', data: []}]},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const pnlChart = new Chart(document.getElementById('pnl_chart'), {
        type: 'line',
        data: {labels: [], datasets: []},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const allocationChart = new Chart(document.getElementById('allocation_chart'), {
        type: 'line',
        data: {labels: [], datasets: []},
        options: {scales:{y:{beginAtZero:true, max:1}}}
    });

    const varChart = new Chart(document.getElementById('var_chart'), {
        type: 'line',
        data: {labels: [], datasets: [{label:'VaR', data:[]}]},
        options: {scales:{y:{beginAtZero:true}}}
    });

    function loadKeypairs() {
        fetch('/keypairs').then(r => r.json()).then(data => {
            const sel = document.getElementById('keypair_select');
            sel.innerHTML = '';
            data.keypairs.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n; opt.textContent = n; sel.appendChild(opt);
            });
            sel.value = data.active || '';
            document.getElementById('active_keypair').textContent = data.active || '';
        });
    }
    document.getElementById('keypair_select').onchange = function() {
        fetch('/keypairs/select', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:this.value})});
    };

    function loadRisk() {
        fetch('/risk').then(r => r.json()).then(data => {
            document.getElementById('risk_tolerance').value = data.risk_tolerance;
            document.getElementById('max_allocation').value = data.max_allocation;
            document.getElementById('risk_multiplier').value = data.risk_multiplier;
        });
    }

    document.getElementById('save_risk').onclick = function() {
        fetch('/risk', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                risk_tolerance: parseFloat(document.getElementById('risk_tolerance').value),
                max_allocation: parseFloat(document.getElementById('max_allocation').value),
                risk_multiplier: parseFloat(document.getElementById('risk_multiplier').value)
            })
        });
    };

    function loadWeights() {
        fetch('/weights').then(r => r.json()).then(data => {
            const div = document.getElementById('weights_controls');
            div.innerHTML = '';
            const labels = [];
            const values = [];
            Object.entries(data).forEach(([name, val]) => {
                const label = document.createElement('label');
                label.textContent = name;
                const inp = document.createElement('input');
                inp.type = 'number';
                inp.step = '0.1';
                inp.value = val;
                inp.dataset.agent = name;
                label.appendChild(inp);
                div.appendChild(label);
                labels.push(name);
                values.push(val);
            });
            weightsChart.data.labels = labels;
            weightsChart.data.datasets[0].data = values;
            weightsChart.update();
        });
    }

    function loadStrategies() {
        fetch('/strategies').then(r => r.json()).then(data => {
            const div = document.getElementById('strategy_controls');
            div.innerHTML = '';
            (data.available || []).forEach(name => {
                const label = document.createElement('label');
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.dataset.strategy = name;
                if(!data.active || data.active.includes(name)) cb.checked = true;
                label.appendChild(cb);
                label.appendChild(document.createTextNode(name));
                div.appendChild(label);
            });
        });
    }

    function loadConfig() {
        fetch('/configs').then(r => r.json()).then(data => {
            document.getElementById('active_config').textContent = data.active || '';
        });
    }

    document.getElementById('save_weights').onclick = function() {
        const data = {};
        document.querySelectorAll('#weights_controls input').forEach(inp => {
            data[inp.dataset.agent] = parseFloat(inp.value);
        });
        fetch('/weights', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify(data)
        });
    };

    document.getElementById('save_strategies').onclick = function() {
        const names = [];
        document.querySelectorAll('#strategy_controls input').forEach(inp => {
            if(inp.checked) names.push(inp.dataset.strategy);
        });
        fetch('/strategies', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify(names)
        });
    };

    function refreshData() {
        fetch('/status').then(r => r.json()).then(data => {
            document.getElementById('status_info').textContent = JSON.stringify(data, null, 2);
        });
        fetch('/positions').then(r => r.json()).then(data => {
            document.getElementById('positions').textContent = JSON.stringify(data, null, 2);
        });
        fetch('/trades').then(r => r.json()).then(data => {
            document.getElementById('trades').textContent = JSON.stringify(data.slice(-10), null, 2);
            const buy = data.filter(t=>t.direction==='buy').length;
            const sell = data.filter(t=>t.direction==='sell').length;
            tradeChart.data.datasets[0].data = [buy, sell];
            tradeChart.update();
        });
        fetch('/roi').then(r => r.json()).then(data => {
            document.getElementById('roi_value').textContent = data.roi.toFixed(4);
            const color = data.roi >= 0 ? 'green' : 'red';
            roiChart.data.labels.push('');
            roiChart.data.datasets[0].data.push(data.roi);
            roiChart.data.datasets[0].borderColor.push(color);
            roiChart.data.datasets[0].backgroundColor.push(color);
            if(roiChart.data.labels.length>50){
                roiChart.data.labels.shift();
                roiChart.data.datasets[0].data.shift();
                roiChart.data.datasets[0].borderColor.shift();
                roiChart.data.datasets[0].backgroundColor.shift();
            }
            roiChart.update();
        });
        fetch('/token_history').then(r => r.json()).then(data => {
            let maxLen = 0;
            Object.values(data).forEach(v => { if(v.pnl_history.length > maxLen) maxLen = v.pnl_history.length; });
            pnlChart.data.labels = Array.from({length:maxLen}, ()=>'');
            allocationChart.data.labels = pnlChart.data.labels;
            Object.entries(data).forEach(([tok, stats]) => {
                let ds = pnlChart.data.datasets.find(d => d.label===tok);
                if(!ds){ ds = {label:tok, data:[]}; pnlChart.data.datasets.push(ds); }
                ds.data = stats.pnl_history;
                let da = allocationChart.data.datasets.find(d => d.label===tok);
                if(!da){ da = {label:tok, data:[]}; allocationChart.data.datasets.push(da); }
                da.data = stats.allocation_history;
            });
            pnlChart.update();
            allocationChart.update();
        });
        fetch('/vars').then(r => r.json()).then(data => {
            document.getElementById('var_values').textContent = JSON.stringify(data.slice(-10), null, 2);
            varChart.data.labels = data.map(()=>'');
            varChart.data.datasets[0].data = data.map(v=>v.value);
            varChart.update();
        });
        fetch('/exposure').then(r => r.json()).then(data => {
            document.getElementById('exposure').textContent = JSON.stringify(data, null, 2);
        });
        fetch('/sharpe').then(r => r.json()).then(data => {
            document.getElementById('sharpe_val').textContent = data.sharpe.toFixed(4);
        });
        fetch('/rl/status').then(r => r.json()).then(data => {
            document.getElementById('rl_status').textContent = JSON.stringify(data);
        });
        fetch('/logs').then(r => r.json()).then(data => {
            const logEl = document.getElementById('logs');
            const newLogs = data.logs.slice(logIndex);
            if(newLogs.length) {
                logEl.textContent += newLogs.join('\n') + '\n';
                logEl.scrollTop = logEl.scrollHeight;
                logIndex = data.logs.length;
            }
        });
        loadWeights();
        loadStrategies();
    }

    loadKeypairs();
    loadConfig();
    loadRisk();
    loadWeights();
    loadStrategies();
    refreshData();
    setInterval(function(){
        refreshData();
        loadConfig();
        loadKeypairs();
        loadStrategies();
    }, 5000);
    try {
        const rlSock = new WebSocket('ws://' + window.location.hostname + ':8767');
        rlSock.onmessage = function(ev) {
            try {
                const data = JSON.parse(ev.data);
                if('loss' in data && 'reward' in data) {
                    document.getElementById('rl_status').textContent = JSON.stringify(data);
                }
            } catch(e) {}
        };
    } catch(e) {}
    try {
        const logSock = new WebSocket('ws://' + window.location.hostname + ':8768');
        logSock.onmessage = function(ev) {
            const el = document.getElementById('logs');
            if(el){
                el.textContent += ev.data + '\n';
                el.scrollTop = el.scrollHeight;
            }
        };
    } catch(e) {}
    </script>
</body>
</html>
"""

STARTUP_PAGE = """
<!doctype html>
<html>
<head>
    <title>Startup Logs</title>
    <style>
        body {font-family: monospace;}
        pre {height: 100vh; overflow-y: scroll;}
    </style>
</head>
<body>
<pre id="logs">{{ logs|join('\n') }}</pre>
<script>
let idx = {{ logs|length }};
function poll() {
    fetch('/logs').then(r => r.json()).then(data => {
        const el = document.getElementById('logs');
        const lines = data.logs.slice(idx);
        if (lines.length) {
            el.textContent += lines.join('\n') + '\n';
            el.scrollTop = el.scrollHeight;
            idx = data.logs.length;
        }
    }).catch(() => {});
}
setInterval(poll, 1000);
</script>
</body>
</html>
"""


@bp.route("/")
def index() -> str:
    return render_template_string(HTML_PAGE)


@bp.route("/startup")
def startup() -> str:
    """Render buffered startup log lines."""
    return render_template_string(STARTUP_PAGE, logs=list(log_buffer))


async def _rl_ws_handler(ws):
    rl_ws_clients.add(ws)
    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        rl_ws_clients.discard(ws)


async def _event_ws_handler(ws, path: str | None = None):
    if path not in (None, "/ws"):
        return
    event_ws_clients.add(ws)
    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        event_ws_clients.discard(ws)


async def _log_ws_handler(ws):
    log_ws_clients.add(ws)
    try:
        for line in list(log_buffer):
            await ws.send(line)
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        log_ws_clients.discard(ws)


def start_websockets() -> dict[str, threading.Thread]:
    """Start websocket servers for RL checkpoints, events and logs.

    Returns a mapping of websocket name to the ``Thread`` running its event
    loop so callers can stop the loops and join the threads for a graceful
    shutdown.
    """
    if websockets is None:
        return {}

    threads: dict[str, threading.Thread] = {}

    def _start_rl_ws() -> None:
        global rl_ws_loop
        rl_ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(rl_ws_loop)
        server = None
        try:
            async def _serve() -> Any:
                return await websockets.serve(
                    _rl_ws_handler,
                    "localhost",
                    8767,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )

            try:
                server = rl_ws_loop.run_until_complete(_serve())
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    logger.error("RL websocket port %s is already in use", 8767)
                else:
                    logger.error(
                        "Failed to start RL websocket on port %s: %s", 8767, e
                    )
                return
            rl_ws_loop.run_forever()
        finally:
            if server is not None:
                server.close()
                rl_ws_loop.run_until_complete(server.wait_closed())
            rl_ws_loop.run_until_complete(rl_ws_loop.shutdown_asyncgens())
            rl_ws_loop.close()
            asyncio.set_event_loop(None)
            rl_ws_loop = None

    def _start_event_ws() -> None:
        global event_ws_loop
        event_ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_ws_loop)
        server = None
        try:
            async def _serve() -> Any:
                return await websockets.serve(
                    _event_ws_handler,
                    "localhost",
                    _EVENT_WS_PORT,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )

            try:
                server = event_ws_loop.run_until_complete(_serve())
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    logger.error(
                        "Event websocket port %s is already in use", _EVENT_WS_PORT
                    )
                else:
                    logger.error(
                        "Failed to start event websocket on port %s: %s",
                        _EVENT_WS_PORT,
                        e,
                    )
                return
            event_ws_loop.run_forever()
        finally:
            if server is not None:
                server.close()
                event_ws_loop.run_until_complete(server.wait_closed())
            event_ws_loop.run_until_complete(event_ws_loop.shutdown_asyncgens())
            event_ws_loop.close()
            asyncio.set_event_loop(None)
            event_ws_loop = None

    def _start_log_ws() -> None:
        global log_ws_loop
        log_ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(log_ws_loop)
        server = None
        try:
            async def _serve() -> Any:
                return await websockets.serve(
                    _log_ws_handler,
                    "localhost",
                    8768,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )

            try:
                server = log_ws_loop.run_until_complete(_serve())
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    logger.error("Log websocket port %s is already in use", 8768)
                else:
                    logger.error(
                        "Failed to start log websocket on port %s: %s", 8768, e
                    )
                return
            log_ws_loop.run_forever()
        finally:
            if server is not None:
                server.close()
                log_ws_loop.run_until_complete(server.wait_closed())
            log_ws_loop.run_until_complete(log_ws_loop.shutdown_asyncgens())
            log_ws_loop.close()
            asyncio.set_event_loop(None)
            log_ws_loop = None

    for name, target in (
        ("rl", _start_rl_ws),
        ("event", _start_event_ws),
        ("log", _start_log_ws),
    ):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        threads[name] = t

    return threads


# Flask application instance populated by ``create_app``.
#
# The application is not created at import time to avoid side effects when the
# module is imported.  Consumers should explicitly call ``create_app`` and
# assign the result to ``ui.app`` if they require a module-level reference.
app: Flask | None = None


if __name__ == "__main__":
    app = create_app()
    threads = start_websockets()
    try:
        app.run()
    finally:
        for loop in (rl_ws_loop, event_ws_loop, log_ws_loop):
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
        for t in threads.values():
            t.join(timeout=1)
        for loop in (rl_ws_loop, event_ws_loop, log_ws_loop):
            if loop is not None:
                loop.close()
        asyncio.run(close_session())
