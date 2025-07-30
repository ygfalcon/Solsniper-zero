import asyncio
import types

import pytest

import solhunter_zero.resource_monitor as rm
from solhunter_zero import event_bus


@pytest.mark.asyncio
async def test_resource_monitor_publish(monkeypatch):
    monkeypatch.setattr(rm.psutil, 'cpu_percent', lambda: 5.0)
    monkeypatch.setattr(rm.psutil, 'virtual_memory', lambda: types.SimpleNamespace(percent=42.0))

    received = []
    unsub = event_bus.subscribe('resource_update', lambda p: received.append(p))

    rm.start_monitor(0.01)
    await asyncio.sleep(0.03)
    rm.stop_monitor()
    unsub()

    assert received
    assert received[0]['cpu'] == 5.0
    assert received[0]['memory'] == 42.0
