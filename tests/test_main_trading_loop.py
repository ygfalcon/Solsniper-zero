"""Integration test for the synchronous ``main`` trading loop.

The trading loop is executed with all external services patched out via
:func:`tests.test_live_trading.setup_live_trading_env`. The function waits up to
``timeout`` seconds for completion and verifies that a mock trade was logged.
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

        assert not thread.is_alive()
        assert trades
