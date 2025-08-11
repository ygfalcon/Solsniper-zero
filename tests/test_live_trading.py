from __future__ import annotations

import sys
import types
from typing import Any, Sequence

from solhunter_zero.config_runtime import Config


def setup_live_trading_env(mp: Any):
    """Prepare a fully in-memory environment for ``main.main``.

    All network bound operations used by :mod:`solhunter_zero.main` are
    replaced with lightweight stand-ins so the trading loop can be executed in
    tests without external side effects.
    """
    # Provide lightweight stubs for modules that require heavy dependencies.
    aiohttp_stub = types.SimpleNamespace(ClientSession=object, TCPConnector=object)
    wallet_stub = types.SimpleNamespace(
        load_selected_keypair=lambda: object(),
        load_keypair=lambda path: object(),
    )
    rich_console = types.SimpleNamespace(Console=lambda *a, **k: None)
    rich_stub = types.SimpleNamespace(console=rich_console)
    mp.setitem(sys.modules, "aiohttp", aiohttp_stub)
    mp.setitem(sys.modules, "solhunter_zero.wallet", wallet_stub)
    mp.setitem(sys.modules, "rich", rich_stub)
    mp.setitem(sys.modules, "rich.console", rich_console)

    from solhunter_zero import main as main_module

    # --- network/IO related patches -------------------------------------
    mp.setattr(main_module.metrics_aggregator, "start", lambda: None)
    mp.setattr(main_module.metrics_aggregator, "publish", lambda *a, **k: None)

    async def _verify_broker_connection(*a: Sequence[Any], **k: Any) -> bool:
        return True

    mp.setattr(main_module.event_bus, "verify_broker_connection", _verify_broker_connection)
    mp.setattr(main_module, "warm_cache", lambda tokens=None: None)
    mp.setattr(main_module, "bootstrap", lambda *a, **k: None)

    async def _place_order_async(*a: Any, **k: Any) -> None:
        return None

    mp.setattr(main_module, "place_order_async", _place_order_async)

    def _perform_startup(config_path, *, offline=False, dry_run=False):
        return {}, Config(), None

    mp.setattr(main_module, "perform_startup", _perform_startup)

    # Avoid background writer threads and capture trades -----------------
    trades: list[dict] = []

    async def _log_trade(self, **kw):  # type: ignore[override]
        trades.append(kw)

    mp.setattr(main_module.Memory, "log_trade", _log_trade)
    mp.setattr(main_module.Memory, "start_writer", lambda self, **kw: None)

    # Minimal strategy/agent managers to keep setup lightweight ----------
    class _DummyStrategyManager:
        def __init__(self, strategies=None):
            pass

        def list_missing(self):
            return []

    class _DummyAgentManager:
        @classmethod
        def from_default(cls):
            return None

        @classmethod
        def from_config(cls, cfg):
            return None

    mp.setattr(main_module, "StrategyManager", _DummyStrategyManager)
    mp.setattr(main_module, "AgentManager", _DummyAgentManager)

    # Trading loop stub that logs a mock trade ---------------------------
    async def _trading_loop(cfg, runtime_cfg, memory, portfolio, state, **kwargs):
        await memory.log_trade(token="MOCK", direction="buy", amount=1.0, price=1.0)
        main_module._first_trade_recorded = True
        return []

    mp.setattr(main_module, "trading_loop", _trading_loop)
    main_module._first_trade_recorded = False

    return main_module, trades
