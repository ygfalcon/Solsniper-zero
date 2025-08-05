import csv
import json
import importlib
import re

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

    # Verify stdout includes start and end capital for each config
    lines = [line for line in captured.out.splitlines() if line.startswith("Config ")]
    assert len(lines) == len(content)
    for line in lines:
        match = re.match(r"Config (\w+): start=([0-9.]+), end=([0-9.]+)", line)
        assert match, f"Unexpected output line: {line}"
        name, start, end = match.groups()
        assert float(start) == 100.0
        expected = next(item["final_capital"] for item in content if item["config"] == name)
        assert abs(float(end) - expected) < 1e-6
