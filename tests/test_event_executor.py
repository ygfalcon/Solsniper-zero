import asyncio
import sys
import types

import pytest

# Provide dummy aiohttp for imports
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

from solhunter_zero.execution import EventExecutor, run_event_loop


def test_event_executor_run(monkeypatch):
    async def fake_stream_depth(token, rate_limit=0.05):
        for _ in range(2):
            yield {"tx_rate": 1.0}

    calls = []

    async def fake_submit(tx, *, socket_path=None, priority_rpc=None):
        calls.append(tx)

    monkeypatch.setattr("solhunter_zero.execution.stream_depth", fake_stream_depth)
    monkeypatch.setattr("solhunter_zero.execution.submit_raw_tx", fake_submit)

    execer = EventExecutor("TOK")

    async def runner():
        await execer.enqueue("A")
        await execer.enqueue("B")
        await execer.run()

    asyncio.run(runner())

    assert calls == ["A", "B"]


def test_run_event_loop_enqueue(monkeypatch):
    captured = []

    async def dummy_run(self):
        await asyncio.sleep(0.05)

    async def dummy_enqueue(self, tx):
        captured.append(tx)

    monkeypatch.setattr(EventExecutor, "run", dummy_run)
    monkeypatch.setattr(EventExecutor, "enqueue", dummy_enqueue)

    async def runner():
        q: asyncio.Queue = asyncio.Queue()
        await q.put(b"X")
        await q.put("Y")
        task = asyncio.create_task(run_event_loop("TOK", q))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(runner())

    assert captured == ["X", "Y"]
