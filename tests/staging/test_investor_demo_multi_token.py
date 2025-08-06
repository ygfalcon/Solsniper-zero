import csv
import json

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(30)
def test_investor_demo_multi_token(tmp_path, monkeypatch):
    """Ensure aggregate summaries cover all tokens in the multi preset."""

    async def fake_arbitrage():
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": [], "profit": 0.0}

    async def fake_flash_loan():
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper():
        investor_demo.used_trade_types.add("sniper")
        return []

    async def fake_dex_scanner():
        investor_demo.used_trade_types.add("dex_scanner")
        return []

    async def fake_route_ffi():
        return {}

    async def fake_jito_stream():
        return []

    def fake_rl_agent():
        return 0.0

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash_loan)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex_scanner)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route_ffi)
    monkeypatch.setattr(investor_demo, "_demo_jito_stream", fake_jito_stream)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", fake_rl_agent)

    investor_demo.main(["--reports", str(tmp_path), "--preset", "multi"])

    json_path = tmp_path / "aggregate_summary.json"
    csv_path = tmp_path / "aggregate_summary.csv"

    assert json_path.exists(), "aggregate_summary.json not found"
    assert csv_path.exists(), "aggregate_summary.csv not found"

    agg = json.loads(json_path.read_text())
    per_token = agg.get("per_token", [])

    csv_rows = list(csv.DictReader(csv_path.open()))
    csv_map = {row["token"]: row for row in csv_rows}
    token_map = {row["token"]: row for row in per_token}

    tokens = set(investor_demo.load_prices(preset="multi").keys())
    assert set(csv_map) == set(token_map) == tokens

    for token in tokens:
        row_csv = csv_map[token]
        row_json = token_map[token]
        assert row_csv["strategy"] == row_json["strategy"]
        for field in ["roi", "sharpe", "final_capital"]:
            assert field in row_csv and field in row_json
            assert float(row_csv[field]) == pytest.approx(float(row_json[field]))
