import csv
import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_one_click_generates_reports(tmp_path: Path, monkeypatch) -> None:
    """Run the demo and verify summary and trade history artifacts."""

    async def fake_arb() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": ["dex1", "dex2"], "profit": 1.0}

    async def fake_flash() -> str:
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper() -> list[str]:
        investor_demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def fake_dex() -> list[str]:
        investor_demo.used_trade_types.add("dex_scanner")
        return ["pool"]

    async def fake_route() -> dict:
        return {"path": ["r1"], "profit": 0.5}

    async def fake_jito() -> list[str]:
        return []

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arb)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route)
    monkeypatch.setattr(investor_demo, "_demo_jito_stream", fake_jito)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)
    investor_demo.used_trade_types = set()

    investor_demo.main(["--reports", str(tmp_path)])

    agg_json = tmp_path / "aggregate_summary.json"
    agg_csv = tmp_path / "aggregate_summary.csv"
    trade_json = tmp_path / "trade_history.json"
    trade_csv = tmp_path / "trade_history.csv"

    for path in [agg_json, agg_csv, trade_json, trade_csv]:
        assert path.exists(), f"{path.name} not found"

    agg_data = json.loads(agg_json.read_text())
    assert {"global_roi", "global_sharpe", "per_token"} <= agg_data.keys()
    assert agg_data["per_token"], "per_token summary is empty"

    with agg_csv.open() as f:
        reader = csv.DictReader(f)
        assert {"token", "strategy", "roi", "sharpe", "final_capital"} == set(
            reader.fieldnames or []
        )

    trades = json.loads(trade_json.read_text())
    assert any(
        {"token", "price"} <= row.keys()
        and (
            {"side", "amount"} <= row.keys()
            or {"action", "capital"} <= row.keys()
        )
        for row in trades
    ), "Trade entry missing required fields"

    with trade_csv.open() as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        assert {"token", "price"} <= headers
        assert ("side" in headers or "action" in headers)
        assert ("amount" in headers or "capital" in headers)

