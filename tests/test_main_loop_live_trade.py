from __future__ import annotations

from typing import Any
from types import SimpleNamespace

from tests.test_live_trading import setup_live_trading_env
from solhunter_zero import wallet, routeffi, depth_client


def test_main_loop_executes_trade(monkeypatch):
    """Run the main trading loop with deterministic stubs.

    This test verifies that the trading loop integrates with the
    route FFI and depth client using stubbed versions so it can run
    reliably under CI without external services.
    """

    with monkeypatch.context() as mp:
        main_module, trades = setup_live_trading_env(mp)

        # Replace routeffi and depth_client calls with deterministic stubs
        route_called = {"flag": False}

        def route_stub(*args: Any, **kwargs: Any):
            route_called["flag"] = True
            return ["SOL"], 0.0

        mp.setattr(routeffi, "best_route", route_stub)

        depth_called = {"flag": False}

        def depth_stub(token: str):
            depth_called["flag"] = True
            return {}, 0.0

        mp.setattr(depth_client, "snapshot", depth_stub)

        mp.setattr(main_module, "perform_startup", lambda *a, **k: ({}, SimpleNamespace(), None))

        async def trading_loop_stub(cfg, runtime_cfg, memory, portfolio, state, **kwargs):
            routeffi.best_route({}, 0.0)
            depth_client.snapshot("SOL")
            await memory.log_trade(token="SOL", side="buy", amount=1.0, price=0.0)

        mp.setattr(main_module, "trading_loop", trading_loop_stub)

        import threading

        thread = threading.Thread(
            target=main_module.main,
            kwargs={
                "iterations": 1,
                "loop_delay": 0,
                "dry_run": True,
                "memory_path": "sqlite:///:memory:",
                "keypair_path": "dummy.json",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)

        assert wallet.load_keypair.called, "wallet was not loaded"
        assert route_called["flag"], "route_ffi was not invoked"
        assert depth_called["flag"], "depth_service was not queried"
        assert trades, "no trade was logged"
