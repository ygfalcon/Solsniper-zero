import json
import sys
import types

import paper

def test_paper_live_flow(tmp_path, monkeypatch):
    fake_crypto = types.ModuleType("cryptography")
    fake_fernet = types.SimpleNamespace(Fernet=object, InvalidToken=Exception)
    fake_crypto.fernet = fake_fernet
    sys.modules.setdefault("cryptography", fake_crypto)
    sys.modules.setdefault("cryptography.fernet", fake_fernet)

    class _BaseModel:
        pass

    class _AnyUrl(str):
        pass

    class _ValidationError(Exception):
        pass

    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def model_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    pydantic_stub = types.SimpleNamespace(
        BaseModel=_BaseModel,
        AnyUrl=_AnyUrl,
        ValidationError=_ValidationError,
        field_validator=field_validator,
        model_validator=model_validator,
    )
    sys.modules.setdefault("pydantic", pydantic_stub)

    from solhunter_zero import wallet, routeffi, depth_client
    flags = {"wallet": False, "route": False, "depth": False}

    def fake_load_keypair(path: str):
        flags["wallet"] = True
        return object()

    async def fake_best_route(*args, **kwargs):
        flags["route"] = True
        return {"path": ["A", "B"], "amount": kwargs.get("amount", 0)}

    async def fake_snapshot(token: str):
        flags["depth"] = True
        return {}, 0.0

    monkeypatch.setattr(wallet, "load_keypair", fake_load_keypair)
    monkeypatch.setattr(routeffi, "best_route", fake_best_route)
    monkeypatch.setattr(depth_client, "snapshot", fake_snapshot)

    ticks = [{"price": 1.0, "timestamp": 1}, {"price": 2.0, "timestamp": 2}]
    dataset = paper._ticks_to_price_file(ticks)
    monkeypatch.setattr(paper, "_fetch_live_dataset", lambda: dataset)

    reports = tmp_path / "reports"
    paper.run(["--live-flow", "--fetch-live", "--reports", str(reports)])

    assert flags["wallet"], "wallet.load_keypair not called"
    assert flags["route"], "routeffi.best_route not called"
    assert flags["depth"], "depth_client.snapshot not called"

    trade_path = reports / "trade_history.json"
    assert trade_path.exists()
    data = json.loads(trade_path.read_text())
    assert data and all(k in data[0] for k in ("token", "side", "amount", "price"))
