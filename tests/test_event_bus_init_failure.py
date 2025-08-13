import asyncio
import importlib
import os

import pytest
@pytest.mark.asyncio
async def test_initialize_event_bus_failure_cancels_tasks(monkeypatch):
    monkeypatch.setenv("BROKER_WS_URLS", "ws://bus")
    import solhunter_zero.event_bus as ev
    ev = importlib.reload(ev)
    import solhunter_zero.config as cfg
    cfg = importlib.reload(cfg)

    async def fake_reachable(urls, timeout=1.0):
        await asyncio.sleep(0)
        return set()

    async def fake_disconnect():
        pass

    monkeypatch.setattr(ev, "_reachable_ws_urls", fake_reachable)
    monkeypatch.setattr(ev, "disconnect_ws", fake_disconnect)

    cfg.initialize_event_bus()
    await asyncio.sleep(0.1)
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert not tasks

    ev.shutdown_event_bus()
    importlib.reload(ev)
