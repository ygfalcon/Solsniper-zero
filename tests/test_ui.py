import time
import os
import io
import json
import pytest
from solders.keypair import Keypair
from solhunter_zero import ui, config
import logging
from collections import deque
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

    for _ in range(100):
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


def test_upload_endpoints_prevent_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(ui.wallet, "KEYPAIR_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(ui.wallet, "ACTIVE_KEYPAIR_FILE", str(tmp_path / "keys" / "active"))
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path / "cfgs"))
    monkeypatch.setattr(config, "ACTIVE_CONFIG_FILE", str(tmp_path / "cfgs" / "active"))
    os.makedirs(ui.wallet.KEYPAIR_DIR, exist_ok=True)
    os.makedirs(config.CONFIG_DIR, exist_ok=True)

    client = ui.app.test_client()

    kp = Keypair()
    data = json.dumps(list(kp.to_bytes()))
    resp = client.post(
        "/keypairs/upload",
        data={"name": "../evil", "file": (io.BytesIO(data.encode()), "kp.json")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert not list((tmp_path / "keys").glob("*.json"))

    resp = client.post(
        "/configs/upload",
        data={"name": "../cfg", "file": (io.BytesIO(b"x"), "c.toml")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert not list((tmp_path / "cfgs").iterdir())


def test_start_auto_selects_single_keypair(monkeypatch, tmp_path):
    monkeypatch.setattr(ui.wallet, "KEYPAIR_DIR", str(tmp_path))
    monkeypatch.setattr(ui.wallet, "ACTIVE_KEYPAIR_FILE", str(tmp_path / "active"))
    os.makedirs(ui.wallet.KEYPAIR_DIR, exist_ok=True)

    kp = Keypair()
    (tmp_path / "only.json").write_text(json.dumps(list(kp.to_bytes())))

    monkeypatch.setattr(ui, "trading_loop", lambda: None)
    monkeypatch.setattr(ui, "load_config", lambda p=None: {})
    monkeypatch.setattr(ui, "apply_env_overrides", lambda c: c)
    monkeypatch.setattr(ui, "set_env_from_config", lambda c: None)
    monkeypatch.setenv("BIRDEYE_API_KEY", "x")
    monkeypatch.setenv("DEX_BASE_URL", "x")

    client = ui.app.test_client()
    resp = client.post("/start")
    assert resp.get_json()["status"] == "started"
    ui.trading_thread.join(timeout=1)
    assert (tmp_path / "active").read_text() == "only"


def test_get_and_set_weights(monkeypatch):
    monkeypatch.delenv("AGENT_WEIGHTS", raising=False)
    client = ui.app.test_client()

    resp = client.get("/weights")
    assert resp.get_json() == {}

    resp = client.post("/weights", json={"sim": 1.2})
    assert resp.get_json()["status"] == "ok"
    assert json.loads(os.getenv("AGENT_WEIGHTS"))["sim"] == 1.2


def test_logs_endpoint(monkeypatch):
    monkeypatch.setattr(ui, "log_buffer", deque(maxlen=5))
    ui.buffer_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().setLevel(logging.INFO)

    logging.getLogger().info("alpha")
    logging.getLogger().error("beta")

    client = ui.app.test_client()
    resp = client.get("/logs")
    logs = resp.get_json()["logs"]

    assert any("alpha" in l for l in logs)
    assert logs[-1] == "beta"


def test_token_history_endpoint(monkeypatch):
    pf = ui.Portfolio(path=None)
    pf.balances = {"tok": Position("tok", 1, 2.0)}
    monkeypatch.setattr(ui, "current_portfolio", pf, raising=False)
    monkeypatch.setattr(ui, "fetch_token_prices", lambda tokens: {"tok": 3.0})
    monkeypatch.setattr(ui, "token_pnl_history", {}, raising=False)
    monkeypatch.setattr(ui, "allocation_history", {}, raising=False)

    client = ui.app.test_client()
    resp = client.get("/token_history")
    data = resp.get_json()
    assert "tok" in data
    assert data["tok"]["pnl_history"][-1] == pytest.approx(1.0)
    assert data["tok"]["allocation_history"][-1] == pytest.approx(1.0)


def _setup_memory(monkeypatch):
    mem = ui.Memory("sqlite:///:memory:")
    monkeypatch.setattr(ui, "Memory", lambda *a, **k: mem)
    return mem


def test_memory_insert(monkeypatch):
    mem = _setup_memory(monkeypatch)
    client = ui.app.test_client()
    resp = client.post(
        "/memory/insert",
        json={
            "sql": "INSERT INTO trades(token,direction,amount,price) VALUES(:t,:d,:a,:p)",
            "params": {"t": "TOK", "d": "buy", "a": 1.0, "p": 2.0},
        },
    )
    assert resp.get_json()["status"] == "ok"
    trades = mem.list_trades()
    assert len(trades) == 1 and trades[0].token == "TOK"


def test_memory_update(monkeypatch):
    mem = _setup_memory(monkeypatch)
    mem.log_trade(token="TOK", direction="buy", amount=1.0, price=2.0)
    client = ui.app.test_client()
    resp = client.post(
        "/memory/update",
        json={"sql": "UPDATE trades SET price=:p WHERE token=:t", "params": {"p": 3.0, "t": "TOK"}},
    )
    assert resp.get_json()["rows"] == 1
    assert mem.list_trades()[0].price == 3.0


def test_memory_query(monkeypatch):
    mem = _setup_memory(monkeypatch)
    mem.log_trade(token="TOK", direction="buy", amount=1.0, price=2.0)
    client = ui.app.test_client()
    resp = client.post(
        "/memory/query",
        json={"sql": "SELECT token, price FROM trades WHERE token=:t", "params": {"t": "TOK"}},
    )
    data = resp.get_json()
    assert data == [{"token": "TOK", "price": 2.0}]


def test_vars_endpoint(monkeypatch):
    mem = _setup_memory(monkeypatch)
    mem.log_var(0.1)
    mem.log_var(0.2)
    client = ui.app.test_client()
    resp = client.get("/vars")
    data = resp.get_json()
    assert [v["value"] for v in data] == [0.1, 0.2]

