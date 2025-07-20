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
    monkeypatch.setattr(ui, "load_config", lambda p=None: {})
    monkeypatch.setattr(ui, "apply_env_overrides", lambda c: c)
    monkeypatch.setattr(ui, "set_env_from_config", lambda c: None)
    monkeypatch.setenv("BIRDEYE_API_KEY", "x")
    monkeypatch.setenv("DEX_BASE_URL", "x")

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
    monkeypatch.setattr(ui, "load_config", lambda p=None: {})
    monkeypatch.setattr(ui, "apply_env_overrides", lambda c: c)
    monkeypatch.setattr(ui, "set_env_from_config", lambda c: None)
    monkeypatch.setattr(ui.wallet, "load_selected_keypair", lambda: None)
    monkeypatch.setattr(ui.wallet, "load_keypair", lambda path: None)
    monkeypatch.setattr(ui, "loop_delay", 0)

    ui.stop_event.clear()

    thread = threading.Thread(target=ui.trading_loop, daemon=True)
    thread.start()
    thread.join(timeout=1)

    assert calls


def test_trading_loop_falls_back_to_env_keypair(monkeypatch):
    used = {}

    async def fake_run_iteration(*args, keypair=None, **kwargs):
        used["keypair"] = keypair
        ui.stop_event.set()

    monkeypatch.setattr(ui.main_module, "_run_iteration", fake_run_iteration)
    monkeypatch.setattr(ui, "Memory", lambda *a, **k: object())
    monkeypatch.setattr(ui, "Portfolio", lambda *a, **k: object())
    monkeypatch.setattr(ui, "load_config", lambda p=None: {})
    monkeypatch.setattr(ui, "apply_env_overrides", lambda c: c)
    monkeypatch.setattr(ui, "set_env_from_config", lambda c: None)
    monkeypatch.setattr(ui.wallet, "load_selected_keypair", lambda: None)

    sentinel = object()

    def fake_load_keypair(path):
        used["path"] = path
        return sentinel

    monkeypatch.setattr(ui.wallet, "load_keypair", fake_load_keypair)
    monkeypatch.setenv("KEYPAIR_PATH", "envpath")
    monkeypatch.setattr(ui, "loop_delay", 0)

    ui.stop_event.clear()
    thread = threading.Thread(target=ui.trading_loop, daemon=True)
    thread.start()
    thread.join(timeout=1)

    assert used["keypair"] is sentinel
    assert used["path"] == "envpath"


def test_get_and_set_risk_params(monkeypatch):
    monkeypatch.delenv("RISK_TOLERANCE", raising=False)
    monkeypatch.delenv("MAX_ALLOCATION", raising=False)
    monkeypatch.delenv("RISK_MULTIPLIER", raising=False)

    client = ui.app.test_client()

    resp = client.get("/risk")
    data = resp.get_json()
    assert "risk_tolerance" in data

    resp = client.post(
        "/risk",
        json={"risk_tolerance": 0.2, "max_allocation": 0.3, "risk_multiplier": 1.5},
    )
    assert resp.get_json()["status"] == "ok"
    assert os.getenv("RISK_TOLERANCE") == "0.2"
    assert os.getenv("MAX_ALLOCATION") == "0.3"
    assert os.getenv("RISK_MULTIPLIER") == "1.5"


def test_get_and_set_discovery_method(monkeypatch):
    monkeypatch.delenv("DISCOVERY_METHOD", raising=False)
    client = ui.app.test_client()

    resp = client.get("/discovery")
    assert resp.get_json()["method"] == "websocket"

    resp = client.post("/discovery", json={"method": "mempool"})
    assert resp.get_json()["status"] == "ok"
    assert os.getenv("DISCOVERY_METHOD") == "mempool"


def test_start_requires_env(monkeypatch):
    monkeypatch.setattr(ui, "trading_loop", lambda: None)
    monkeypatch.setattr(ui, "load_config", lambda p=None: {})
    monkeypatch.setattr(ui, "apply_env_overrides", lambda c: c)
    monkeypatch.setattr(ui, "set_env_from_config", lambda c: None)
    monkeypatch.delenv("BIRDEYE_API_KEY", raising=False)
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    monkeypatch.delenv("DEX_BASE_URL", raising=False)
    client = ui.app.test_client()
    resp = client.post("/start")
    assert resp.status_code == 400
    msg = resp.get_json()["message"]
    assert "DEX_BASE_URL" in msg
    assert "BIRDEYE_API_KEY or SOLANA_RPC_URL" in msg

