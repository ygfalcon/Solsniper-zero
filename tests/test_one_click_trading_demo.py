import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_aggregated_summary_exists(tmp_path: Path, monkeypatch, capsys) -> None:
    """Ensure aggregated metrics file is created with key statistics."""

    async def fake_arbitrage() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": [], "profit": 0.0}

    async def fake_flash() -> str:
        investor_demo.used_trade_types.add("flash_loan")
        return "demo_sig"

    async def fake_sniper() -> list[str]:
        investor_demo.used_trade_types.add("sniper")
        return []

    async def fake_scanner() -> list[str]:
        investor_demo.used_trade_types.add("dex_scanner")
        return []

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_scanner)

    investor_demo.main(["--reports", str(tmp_path)])

    out = capsys.readouterr().out
    assert "Wrote reports to" in out
    assert "Strategy" in out

    agg_path = tmp_path / "aggregated_summary.json"
    assert agg_path.exists(), "aggregated_summary.json not found"

    data = json.loads(agg_path.read_text())
    for key in ["total_roi", "average_sharpe", "best_strategy", "worst_strategy"]:
        assert key in data
