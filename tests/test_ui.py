import time
from solhunter_zero import ui


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


def test_balances_endpoint(monkeypatch):
    from solhunter_zero.portfolio import Portfolio, Position

    pf = Portfolio(path=None)
    pf.balances["tok"] = Position("tok", 2, 1.0)

    monkeypatch.setattr(ui, "current_portfolio", pf)

    client = ui.app.test_client()
    resp = client.get("/balances")
    data = resp.get_json()
    assert data["tok"]["amount"] == 2
