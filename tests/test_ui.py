import time
import os
from solders.keypair import Keypair
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


def test_keypair_add_and_select(monkeypatch, tmp_path):
    monkeypatch.setattr(ui.wallet, "KEYPAIR_DIR", str(tmp_path))
    monkeypatch.setattr(ui.wallet, "ACTIVE_KEYPAIR_FILE", str(tmp_path / "active"))
    os.makedirs(ui.wallet.KEYPAIR_DIR, exist_ok=True)

    client = ui.app.test_client()

    kp = Keypair()
    resp = client.post("/keypairs", json={"name": "kp1", "keypair": list(kp.to_bytes())})
    assert resp.get_json()["status"] == "saved"

    data = client.get("/keypairs").get_json()
    assert "kp1" in data["keypairs"]
    assert data["active"] is None

    resp = client.post("/keypairs/select", json={"name": "kp1"})
    assert resp.get_json()["status"] == "selected"
    data = client.get("/keypairs").get_json()
    assert data["active"] == "kp1"

    loaded = ui.wallet.load_selected_keypair()
    assert loaded.to_bytes() == kp.to_bytes()
