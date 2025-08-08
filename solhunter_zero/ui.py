import threading
import os
import asyncio
import json
from dataclasses import asdict, is_dataclass
import logging
import socket
from collections import deque
from typing import Any
import time
import subprocess
import sys
from pathlib import Path

from flask import Flask, Blueprint, jsonify, request, render_template

from .http import close_session
from .util import install_uvloop
from .event_bus import subscription, publish
try:
    import websockets
except Exception:  # pragma: no cover - optional
    websockets = None

import sqlalchemy as sa
import numpy as np

from .config import (
    load_config,
    apply_env_overrides,
    set_env_from_config,
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

# websocket ping configuration
_WS_PING_INTERVAL = float(os.getenv("WS_PING_INTERVAL", "20") or 20)
_WS_PING_TIMEOUT = float(os.getenv("WS_PING_TIMEOUT", "20") or 20)

bp = Blueprint("ui", __name__)

_DEFAULT_PRESET = Path(__file__).resolve().parent.parent / "config" / "default.toml"

# in-memory log storage for UI access (initialised in ``create_app``)
log_buffer: deque[str] = deque()
buffer_handler: logging.Handler | None = None
_SUBSCRIPTIONS: list[Any] = []

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

# currently active portfolio and keypair used by the trading loop
current_portfolio: Portfolio | None = None
current_keypair = None
pnl_history: list[float] = []
token_pnl_history: dict[str, list[float]] = {}
allocation_history: dict[str, list[float]] = {}

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

    # auto-select single keypair and configuration on startup
    try:
        if wallet.get_active_keypair_name() is None:
            keys = wallet.list_keypairs()
            if len(keys) == 1:
                wallet.select_keypair(keys[0])
                if not os.getenv("KEYPAIR_PATH"):
                    os.environ["KEYPAIR_PATH"] = os.path.join(
                        wallet.KEYPAIR_DIR, keys[0] + ".json"
                    )
    except Exception as exc:
        print(
            f"Wallet interaction failed: {exc}\n"
            "Run 'solhunter-wallet' manually or set the MNEMONIC environment variable.",
            file=sys.stderr,
        )

    if get_active_config_name() is None:
        configs = list_configs()
        if len(configs) == 1:
            select_config(configs[0])
            set_env_from_config(load_selected_config())

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
    ]
    for sub in _SUBSCRIPTIONS:
        sub.__enter__()

    return app


async def trading_loop(memory: BaseMemory | None = None) -> None:
    global current_portfolio, current_keypair

    cfg = apply_env_overrides(load_config("config.toml"))
    set_env_from_config(cfg)

    memory = memory or Memory("sqlite:///memory.db")
    portfolio = Portfolio()

    current_portfolio = portfolio
    set_env_from_config(load_selected_config())
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
        current_keypair = keypair
        await main_module._run_iteration(
            memory,
            portfolio,
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

    try:
        from . import data_sync

        asyncio.run(data_sync.sync_recent())
    except Exception as exc:  # pragma: no cover - ignore sync errors
        logging.getLogger(__name__).warning("data sync failed: %s", exc)

    # auto-select the only available keypair if none is active
    try:
        if wallet.get_active_keypair_name() is None:
            keys = wallet.list_keypairs()
            if len(keys) == 1:
                wallet.select_keypair(keys[0])
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
    start_all_proc.wait()


@bp.route("/start_all", methods=["POST"])
def start_all_route() -> dict:
    global start_all_thread
    if start_all_thread and start_all_thread.is_alive():
        return jsonify({"status": "already running"})
    start_all_thread = threading.Thread(target=_run_start_all, daemon=True)
    start_all_thread.start()
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
    if request.method == "POST":
        data = request.get_json() or {}
        rt = data.get("risk_tolerance")
        ma = data.get("max_allocation")
        rm = data.get("risk_multiplier")
        if rt is not None:
            os.environ["RISK_TOLERANCE"] = str(rt)
        if ma is not None:
            os.environ["MAX_ALLOCATION"] = str(ma)
        if rm is not None:
            os.environ["RISK_MULTIPLIER"] = str(rm)
            publish("risk_updated", {"multiplier": float(rm)})
        return jsonify({"status": "ok"})
    return jsonify(
        {
            "risk_tolerance": float(os.getenv("RISK_TOLERANCE", "0.1")),
            "max_allocation": float(os.getenv("MAX_ALLOCATION", "0.2")),
            "risk_multiplier": float(os.getenv("RISK_MULTIPLIER", "1.0")),
        }
    )


@bp.route("/weights", methods=["GET", "POST"])
def agent_weights() -> dict:
    """Get or update agent weighting factors."""
    if request.method == "POST":
        weights = request.get_json() or {}
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
    if request.method == "POST":
        method = (request.get_json() or {}).get("method")
        if method:
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
    name = request.form.get("name") or (file.filename if file else None)
    if not file or not name:
        return jsonify({"error": "missing file or name"}), 400
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
    name = request.form.get("name") or (file.filename if file else None)
    if not file or not name:
        return jsonify({"error": "missing file or name"}), 400
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
    mem = Memory("sqlite:///memory.db")
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
    mem = Memory("sqlite:///memory.db")
    data = [
        {"value": v.value, "timestamp": v.timestamp.isoformat()}
        for v in mem.list_vars()[-50:]
    ]
    return jsonify(data)


@bp.route("/roi")
def roi() -> dict:
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
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    entry = sum(p.amount * p.entry_price for p in pf.balances.values())
    value = sum(
        p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items()
    )
    pnl = value - entry
    pnl_history.append(pnl)
    return jsonify({"pnl": pnl, "history": pnl_history})


@bp.route("/token_history")
def token_history() -> dict:
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    total = pf.total_value(prices)
    result: dict[str, dict[str, list[float]]] = {}
    for token, pos in pf.balances.items():
        price = prices.get(token, pos.entry_price)
        pnl = (price - pos.entry_price) * pos.amount
        token_pnl_history.setdefault(token, []).append(pnl)
        alloc = (pos.amount * price) / total if total else 0.0
        allocation_history.setdefault(token, []).append(alloc)
        result[token] = {
            "pnl_history": token_pnl_history[token],
            "allocation_history": allocation_history[token],
        }
    return jsonify(result)


@bp.route("/balances")
def balances() -> dict:
    """Return portfolio balances with USD values."""
    pf = Portfolio()
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
    return jsonify(
        {
            "trading_loop": trading_alive,
            "rl_daemon": rl_alive,
            "depth_service": depth_alive,
            "event_bus": event_alive,
            "heartbeat": heartbeat_alive,
            "system_metrics": system_metrics,
        }
    )


@bp.route("/memory/insert", methods=["POST"])
def memory_insert() -> dict:
    data = request.get_json() or {}
    sql = data.get("sql")
    params = data.get("params", {})
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    mem = Memory("sqlite:///memory.db")
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        session.commit()
    return jsonify({"status": "ok", "rows": result.rowcount})


@bp.route("/memory/update", methods=["POST"])
def memory_update() -> dict:
    data = request.get_json() or {}
    sql = data.get("sql")
    params = data.get("params", {})
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    mem = Memory("sqlite:///memory.db")
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        session.commit()
    return jsonify({"status": "ok", "rows": result.rowcount})


@bp.route("/memory/query", methods=["POST"])
def memory_query() -> dict:
    data = request.get_json() or {}
    sql = data.get("sql")
    params = data.get("params", {})
    if not sql:
        return jsonify({"error": "missing sql"}), 400
    mem = Memory("sqlite:///memory.db")
    with mem.Session() as session:
        result = session.execute(sa.text(sql), params)
        rows = [dict(row._mapping) for row in result]
    return jsonify(rows)


@bp.route("/")
def index() -> str:
    return render_template("index.html")


async def _rl_ws_handler(ws):
    rl_ws_clients.add(ws)
    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        rl_ws_clients.discard(ws)


async def _event_ws_handler(ws):
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


if __name__ == "__main__":
    app = create_app()
    if websockets is not None:
        def _start_rl_ws():
            global rl_ws_loop
            rl_ws_loop = asyncio.new_event_loop()
            rl_ws_loop.run_until_complete(
                websockets.serve(
                    _rl_ws_handler,
                    "localhost",
                    8767,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )
            )
            rl_ws_loop.run_forever()

        def _start_event_ws():
            global event_ws_loop
            event_ws_loop = asyncio.new_event_loop()
            event_ws_loop.run_until_complete(
                websockets.serve(
                    _event_ws_handler,
                    "localhost",
                    8766,
                    path="/ws",
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )
            )
            event_ws_loop.run_forever()

        def _start_log_ws():
            global log_ws_loop
            log_ws_loop = asyncio.new_event_loop()
            log_ws_loop.run_until_complete(
                websockets.serve(
                    _log_ws_handler,
                    "localhost",
                    8768,
                    ping_interval=_WS_PING_INTERVAL,
                    ping_timeout=_WS_PING_TIMEOUT,
                )
            )
            log_ws_loop.run_forever()

        threading.Thread(target=_start_rl_ws, daemon=True).start()
        threading.Thread(target=_start_event_ws, daemon=True).start()
        threading.Thread(target=_start_log_ws, daemon=True).start()

    try:
        app.run()
    finally:
        asyncio.run(close_session())
