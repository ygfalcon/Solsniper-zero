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

    # Plot files are produced when matplotlib is installed
    if importlib.util.find_spec("matplotlib") is not None:
        for name in ["buy_hold", "momentum", "mixed"]:
            assert (tmp_path / f"{name}.png").exists(), f"Missing plot {name}.png"
