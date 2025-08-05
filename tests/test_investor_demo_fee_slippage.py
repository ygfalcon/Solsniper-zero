import json
from pathlib import Path

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(30)
def test_fee_slippage_reduces_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = Path(__file__).parent / "data" / "prices_short.json"

    async def fake_arbitrage() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {}

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")
        return None

    async def fake_sniper() -> list:
        investor_demo.used_trade_types.add("sniper")
        return []

    async def fake_dex() -> list:
        investor_demo.used_trade_types.add("dex_scanner")
        return []

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)

    base_reports = tmp_path / "baseline"
    fee_reports = tmp_path / "fees"

    investor_demo.main(["--reports", str(base_reports), "--data", str(data_path)])
    investor_demo.main([
        "--reports",
        str(fee_reports),
        "--data",
        str(data_path),
        "--fee",
        "0.01",
        "--slippage",
        "0.01",
    ])

    summary1 = json.loads((base_reports / "summary.json").read_text())
    summary2 = json.loads((fee_reports / "summary.json").read_text())

    roi1 = {row["config"]: row["roi"] for row in summary1}
    roi2 = {row["config"]: row["roi"] for row in summary2}
    cap1 = {row["config"]: row["final_capital"] for row in summary1}
    cap2 = {row["config"]: row["final_capital"] for row in summary2}

    assert roi1.keys() == roi2.keys()
    assert cap1.keys() == cap2.keys()

    for name in roi1:
        assert roi2[name] < roi1[name]
        assert cap2[name] < cap1[name]
