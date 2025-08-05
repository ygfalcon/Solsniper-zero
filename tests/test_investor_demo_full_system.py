import asyncio
import json
import importlib
import importlib.util
import re
import sys
import threading
from pathlib import Path

import pytest

from solhunter_zero import investor_demo
from solhunter_zero.event_bus import subscribe


def test_investor_demo_full_system(tmp_path, monkeypatch, capsys, dummy_mem):
    """Ensure full-system mode uses run_rl_demo and records reward metrics."""

    reward = 5.0
    called = {"value": False}

    def stub_run_rl_demo(report_dir: Path) -> float:
        called["value"] = True
        metrics_file = Path(report_dir) / "rl_metrics.json"
        metrics_file.write_text(json.dumps({"loss": [0.0], "rewards": [reward]}))
        return reward

    monkeypatch.setattr(investor_demo, "run_rl_demo", stub_run_rl_demo)
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", lambda w, c: w)

    investor_demo.main(["--full-system", "--reports", str(tmp_path)])

    assert called["value"], "run_rl_demo was not invoked"

    metrics_path = tmp_path / "rl_metrics.json"
    assert metrics_path.exists()

    out = capsys.readouterr().out
    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade results not printed"
    results = json.loads(match.group(1))
    assert results.get("rl_reward") == reward
    assert f'"rl_reward": {reward}' in out

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == reward


@pytest.mark.asyncio
async def test_investor_demo_price_streams(tmp_path, monkeypatch):
    sys.modules.pop("websockets", None)
    spec = importlib.util.find_spec("websockets")
    assert spec and spec.loader
    websockets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(websockets)
    sys.modules["websockets"] = websockets

    import solhunter_zero.price_stream_manager as psm
    import solhunter_zero.investor_demo as demo_mod
    importlib.reload(psm)
    investor_demo = importlib.reload(demo_mod)

    closed1 = asyncio.Event()
    closed2 = asyncio.Event()

    async def orca_handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"token": "TOK", "price": 1.0}))
        await ws.wait_closed()
        closed1.set()

    async def ray_handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"token": "TOK", "price": 2.0}))
        await ws.wait_closed()
        closed2.set()

    server1 = await websockets.serve(orca_handler, "localhost", 0)
    port1 = server1.sockets[0].getsockname()[1]
    server2 = await websockets.serve(ray_handler, "localhost", 0)
    port2 = server2.sockets[0].getsockname()[1]

    events = []
    unsub = subscribe("price_update", lambda p: events.append(p))

    threads: list[threading.Thread] = []
    orig_thread = threading.Thread

    class RecordingThread(orig_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

    monkeypatch.setattr(investor_demo.threading, "Thread", RecordingThread)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        investor_demo.main,
        [
            "--price-streams",
            f"orca=ws://localhost:{port1},raydium=ws://localhost:{port2}",
            "--tokens",
            "TOK",
            "--reports",
            str(tmp_path),
        ],
    )

    unsub()

    await asyncio.sleep(0.1)
    assert any(e.get("venue") == "orca" for e in events)
    assert any(e.get("venue") == "raydium" for e in events)
    assert closed1.is_set()
    assert closed2.is_set()
    for t in threads:
        t.join(timeout=1)

    server1.close()
    await server1.wait_closed()
    server2.close()
    await server2.wait_closed()
