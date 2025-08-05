import json

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(30)
def test_investor_demo_full_system(tmp_path, monkeypatch, dummy_mem):
    """Run full-system mode and record deterministic trade outputs."""

    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)

    async def fake_arbitrage() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": ["dex1", "dex2"], "profit": 1.0}

    async def fake_flash_loan() -> str:
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper() -> list[str]:
        investor_demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def fake_dex_scanner() -> list[str]:
        investor_demo.used_trade_types.add("dex_scanner")
        return ["pool1"]

    async def fake_route_ffi() -> dict:
        investor_demo.used_trade_types.add("route_ffi")
        return {"path": ["r1", "r2"], "profit": 2.0}

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash_loan)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex_scanner)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route_ffi)

    investor_demo.main(["--full-system", "--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights["arbitrage_path"] == ["dex1", "dex2"]
    assert highlights["arbitrage_profit"] == 1.0
    assert highlights["flash_loan_signature"] == "sig"
    assert highlights["sniper_tokens"] == ["TKN"]
    assert highlights["dex_new_pools"] == ["pool1"]
    assert highlights["route_ffi_path"] == ["r1", "r2"]
    assert highlights["route_ffi_profit"] == 2.0

    assert investor_demo.used_trade_types == {
        "arbitrage",
        "flash_loan",
        "sniper",
        "dex_scanner",
        "route_ffi",
    }


@pytest.mark.timeout(30)
def test_investor_demo_missing_trade_type(tmp_path, monkeypatch, dummy_mem):
    """Missing trade types should raise a runtime error."""

    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)

    async def fake_arbitrage() -> dict:
        return {"path": [], "profit": 0.0}

    async def fake_flash_loan() -> str:
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper() -> list[str]:
        investor_demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def fake_dex_scanner() -> list[str]:
        investor_demo.used_trade_types.add("dex_scanner")
        return ["pool1"]

    async def fake_route_ffi() -> dict:
        investor_demo.used_trade_types.add("route_ffi")
        return {"path": ["r1", "r2"], "profit": 2.0}

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash_loan)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex_scanner)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route_ffi)

    with pytest.raises(RuntimeError) as exc:
        investor_demo.main(["--full-system", "--reports", str(tmp_path)])

    msg = str(exc.value)
    assert "arbitrage" in msg
