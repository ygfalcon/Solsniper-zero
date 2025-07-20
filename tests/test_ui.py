import time
import os
from solders.keypair import Keypair
from solhunter_zero import ui
from solhunter_zero.portfolio import Position
import threading


def test_start_and_stop(monkeypatch):
    calls = []

    def fake_loop():
        calls.append(True)
        ui.stop_event.set()

    monkeypatch.setattr(ui, "trading_loop", fake_loop)
    monkeypatch.setattr(ui, "loop_delay", 0)

    client = ui.app.test_client()

    resp = client.post("/start")
    assert resp.get_json()["status"] == "started"

    for _ in range(500):
        time.sleep(0.01)
        if calls:
            break

    resp = client.post("/start")
    assert resp.get_json()["status"] in {"already running", "started"}

    resp = client.post("/stop")
    assert resp.get_json()["status"] == "stopped"

    ui.trading_thread.join(timeout=1)
    assert not ui.trading_thread.is_alive()
    assert calls



def test_balances_includes_usd(monkeypatch):
    pf = ui.Portfolio(path=None)
    pf.balances = {"tok": Position("tok", 2, 1.0)}

    monkeypatch.setattr(ui, "Portfolio", lambda *a, **k: pf)
    monkeypatch.setattr(ui, "fetch_token_prices", lambda tokens: {"tok": 3.0})


    client = ui.app.test_client()
    resp = client.get("/balances")
    data = resp.get_json()


    assert data["tok"]["price"] == 3.0
    assert data["tok"]["usd"] == 6.0
    assert data["tok"]["amount"] == 2


def test_trading_loop_awaits_run_iteration(monkeypatch):
    calls = []

    async def fake_run_iteration(*args, **kwargs):
        calls.append(True)
        ui.stop_event.set()

    monkeypatch.setattr(ui.main_module, "_run_iteration", fake_run_iteration)
    monkeypatch.setattr(ui, "Memory", lambda *a, **k: object())
    monkeypatch.setattr(ui, "Portfolio", lambda *a, **k: object())
    monkeypatch.setattr(ui.wallet, "load_selected_keypair", lambda: None)
    monkeypatch.setattr(ui.wallet, "load_keypair", lambda path: None)
    monkeypatch.setattr(ui, "loop_delay", 0)

    ui.stop_event.clear()

    thread = threading.Thread(target=ui.trading_loop, daemon=True)
    thread.start()
    thread.join(timeout=1)

    assert calls

