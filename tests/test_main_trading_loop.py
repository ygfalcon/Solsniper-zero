"""Integration-like test for the synchronous `main` trading loop.

This test launches the trading loop with all network and exchange layers
mocked using utilities from :mod:`tests.test_live_trading`. It waits up to
``timeout`` seconds for a trade to be recorded and verifies that either a trade
has been logged or the ``_first_trade_recorded`` flag has been set.

Expected runtime: <2s. The test relies on stubbed dependencies and runs in
dry-run mode to avoid external network calls.
"""

from __future__ import annotations

import threading

from tests.test_live_trading import setup_live_trading_env


def test_main_loop_records_trade(monkeypatch):
    timeout = 2
    with monkeypatch.context() as mp:
        main_module, trades = setup_live_trading_env(mp)

        thread = threading.Thread(
            target=main_module.main,
            kwargs={
                "iterations": 1,
                "dry_run": True,
                "loop_delay": 0,
                "memory_path": "sqlite:///:memory:",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)

        assert trades or main_module._first_trade_recorded
