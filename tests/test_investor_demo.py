import csv
import json
import importlib

from solhunter_zero import investor_demo


def test_investor_demo(tmp_path, capsys):
    investor_demo.main(
        [
            "--reports",
            str(tmp_path),
            "--capital",
            "100",
        ]
    )

    captured = capsys.readouterr()
    assert "Top performer" in captured.out

    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    trade_csv = tmp_path / "trade_history.csv"
    highlights_json = tmp_path / "highlights.json"

    assert summary_json.exists(), "Summary JSON not generated"
    assert summary_csv.exists(), "Summary CSV not generated"
    assert trade_csv.exists(), "Trade history CSV not generated"
    assert highlights_json.exists(), "Highlights JSON not generated"

    content = json.loads(summary_json.read_text())
    csv_rows = list(csv.DictReader(summary_csv.open()))
    trade_rows = list(csv.DictReader(trade_csv.open()))

    # JSON and CSV summaries should contain the same number of entries
    assert len(content) == len(csv_rows)

    # Expect results for all configured strategies
    configs = {entry.get("config") for entry in content}
    assert {"buy_hold", "momentum", "mixed"} <= configs

    # Each summary entry must include core metrics
    for entry in content:
        for key in ["roi", "sharpe", "drawdown", "final_capital"]:
            assert key in entry

    # Trade history should have entries for multiple periods
    assert trade_rows, "Trade history should contain rows"
    assert {row["config"] for row in trade_rows} >= {"buy_hold", "momentum", "mixed"}

    # Highlights should identify highest ROI and largest drawdown
    highlights = json.loads(highlights_json.read_text())
    assert "highest_roi" in highlights and "largest_drawdown" in highlights

    # Plot files are produced when matplotlib is installed
    if importlib.util.find_spec("matplotlib") is not None:
        for name in ["buy_hold", "momentum", "mixed"]:
            assert (tmp_path / f"{name}.png").exists(), f"Missing plot {name}.png"

    # Demo should exercise arbitrage and flash loan trade types
    assert {"arbitrage", "flash_loan"} <= investor_demo.used_trade_types
