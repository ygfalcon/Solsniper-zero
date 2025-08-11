from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import solhunter_zero.main as main_module
from solhunter_zero import wallet, routeffi, depth_client
from solhunter_zero import loop as loop_module
from solhunter_zero import event_bus
from solhunter_zero.agents.discovery import DiscoveryAgent


def setup_live_trading_env(monkeypatch):
    """Prepare a lightweight trading environment for live loop tests.

    Parameters
    ----------
    monkeypatch:
        Pytest fixture used to apply temporary patches.

    Returns
    -------
    tuple
        The patched ``main`` module and a mutable list capturing logged trades.
    """

    # Required environment variables for the trading loop
    monkeypatch.setenv("SOLANA_RPC_URL", "http://localhost:8899")
    monkeypatch.setenv("KEYPAIR_PATH", "dummy.json")
    monkeypatch.setenv("USE_DEPTH_STREAM", "0")
    monkeypatch.setenv("AGENTS", "")

    # Stub keypair loading to avoid disk or wallet interactions
    def fake_load_keypair(_path: str) -> SimpleNamespace:
        fake_load_keypair.called = True
        return SimpleNamespace()

    fake_load_keypair.called = False
    monkeypatch.setattr(wallet, "load_keypair", fake_load_keypair)

    # Wrap heavy external dependencies to record invocation without side effects
    route_calls: list[tuple[tuple, dict]] = []

    def route_wrapper(*args: Any, **kwargs: Any):
        route_calls.append((args, kwargs))
        return ["SOL"], 0.0

    monkeypatch.setattr(routeffi, "best_route", route_wrapper)

    depth_calls: list[str] = []

    def depth_wrapper(token: str):
        depth_calls.append(token)
        return {}, 0.0

    monkeypatch.setattr(depth_client, "snapshot", depth_wrapper)

    # Patch token discovery to return a static list
    async def discover_stub(self, *args: Any, **kwargs: Any):
        return ["SOL"]

    monkeypatch.setattr(DiscoveryAgent, "discover_tokens", discover_stub)
    # Expose patched agent via main module for compatibility
    monkeypatch.setattr(main_module, "DiscoveryAgent", DiscoveryAgent, raising=False)

    # Dummy strategy manager returning a basic buy action
    class DummyStrategyManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            pass

        async def evaluate(self, token: str, portfolio: Any, **kwargs: Any):
            routeffi.best_route({}, 0.0)
            depth_client.snapshot(token)
            return [{"token": token, "side": "buy", "amount": 1.0, "price": 0.0}]

        def decide(self, token: str, _pred: Any, _prices: Any) -> str:
            # Simple decision logic used by the trading loop
            return "buy"

    monkeypatch.setattr(main_module, "StrategyManager", DummyStrategyManager)

    # Avoid initializing heavy agent manager machinery
    monkeypatch.setattr(
        main_module.AgentManager,
        "from_default",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        main_module.AgentManager,
        "from_config",
        classmethod(lambda cls, cfg: None),
    )

    # Short-circuit order placement to avoid network calls and mark first trade
    async def fake_place_order_async(*args: Any, **kwargs: Any):
        loop_module._first_trade_recorded = True
        main_module._first_trade_recorded = True
        return {"dry_run": True}

    monkeypatch.setattr(loop_module, "place_order_async", fake_place_order_async)
    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    # Simplify event bus interactions
    monkeypatch.setattr(event_bus, "get_event_bus_url", lambda: "ws://dummy", raising=False)
    monkeypatch.setattr(event_bus, "start_ws_server", lambda: None, raising=False)
    monkeypatch.setattr(event_bus, "stop_ws_server", lambda: None, raising=False)

    async def _verify_broker_connection():
        return True

    monkeypatch.setattr(
        event_bus, "verify_broker_connection", _verify_broker_connection, raising=False
    )
    monkeypatch.setattr(event_bus, "subscribe", lambda *a, **k: lambda: None, raising=False)

    # Capture trades logged through Memory
    trades: list[dict[str, Any]] = []

    async def log_trade_stub(self, *args: Any, **kwargs: Any) -> int:
        trades.append(kwargs)
        loop_module._first_trade_recorded = True
        main_module._first_trade_recorded = True
        return 1

    monkeypatch.setattr(main_module.Memory, "log_trade", log_trade_stub)

    # Ensure first trade flag starts cleared
    loop_module._first_trade_recorded = False
    main_module._first_trade_recorded = False

    return main_module, trades


__all__ = ["setup_live_trading_env"]
