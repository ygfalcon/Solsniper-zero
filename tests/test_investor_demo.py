import csv
import json
import importlib

import pytest

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
    assert trade_csv.exists(), "Trade history CSV not generated"
    assert trade_json.exists(), "Trade history JSON not generated"

    trade_rows = list(csv.DictReader(trade_csv.open()))
    trade_data = json.loads(trade_json.read_text())
    assert trade_rows, "Trade history CSV empty"
    assert trade_data, "Trade history JSON empty"
    assert all("capital" in r for r in trade_rows)
    # Should record capital for periods beyond the starting point
    assert any(int(r["period"]) > 0 for r in trade_rows)

    # Verify final capital in trade history matches summary
    final_caps: dict[str, float] = {}
    for record in trade_data:
        final_caps[record["strategy"]] = record["capital"]
    for entry in content:
        strat = entry["config"]
        assert strat in final_caps
        assert entry["final_capital"] == pytest.approx(final_caps[strat])

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
