import csv
import json
import importlib

from solhunter_zero import investor_demo


def test_investor_demo(tmp_path):
    investor_demo.main(
        [
            "--reports",
            str(tmp_path),
            "--capital",
            "100",
        ]
    )

    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"

    assert summary_json.exists(), "Summary JSON not generated"
    assert summary_csv.exists(), "Summary CSV not generated"

    content = json.loads(summary_json.read_text())
    csv_rows = list(csv.DictReader(summary_csv.open()))

    # JSON and CSV summaries should contain the same number of entries
    assert len(content) == len(csv_rows)

    # Expect results for all configured strategies
    configs = {entry.get("config") for entry in content}
    assert {"buy_hold", "momentum", "mixed"} <= configs

    # Each summary entry must include core metrics
    for entry in content:
        for key in ["roi", "sharpe", "drawdown", "final_capital"]:
            assert key in entry

    # Trade history and highlight files should be generated
    trade_csv = tmp_path / "trade_history.csv"
    trade_json = tmp_path / "trade_history.json"
    assert trade_csv.exists() or trade_json.exists(), "Trade history not generated"
    if trade_csv.exists():
        trade_rows = list(csv.DictReader(trade_csv.open()))
        assert trade_rows, "Trade history CSV empty"
        assert all("capital" in r for r in trade_rows)
        # Should record capital for periods beyond the starting point
        assert any(int(r["period"]) > 0 for r in trade_rows)
    else:
        trade_data = json.loads(trade_json.read_text())
        assert trade_data, "Trade history JSON empty"
        assert all("capital" in r for r in trade_data)
        assert any(r["period"] > 0 for r in trade_data)

    highlights_path = tmp_path / "highlights.json"
    assert highlights_path.exists(), "Highlights JSON not generated"
    highlights = json.loads(highlights_path.read_text())
    assert highlights, "Highlights JSON empty"

    # Highlight metrics should identify the top performing strategy
    top_summary = max(content, key=lambda e: e["final_capital"])
    assert highlights.get("top_strategy") == top_summary["config"]
    assert highlights.get("top_final_capital") == top_summary["final_capital"]

    # Plot files are produced when matplotlib is installed
    if importlib.util.find_spec("matplotlib") is not None:
        for name in ["buy_hold", "momentum", "mixed"]:
            assert (tmp_path / f"{name}.png").exists(), f"Missing plot {name}.png"

    # Demo should exercise arbitrage and flash loan trade types
    assert {"arbitrage", "flash_loan"} <= investor_demo.used_trade_types


def test_used_trade_types_reset(tmp_path, monkeypatch):
    investor_demo.used_trade_types.add("legacy")

    seen_before: list[set[str]] = []

    async def fake_arbitrage() -> None:
        seen_before.append(set(investor_demo.used_trade_types))
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)

    investor_demo.main(["--reports", str(tmp_path)])

    assert seen_before == [set()]
    assert investor_demo.used_trade_types == {"arbitrage", "flash_loan"}
