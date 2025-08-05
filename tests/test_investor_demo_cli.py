import os
import shutil
import subprocess
import sys
from pathlib import Path
import csv

import pytest

from solhunter_zero import investor_demo


def _run_and_check(cmd: list[str], reports_dir: Path, repo_root: Path, env: dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Wrote reports" in result.stdout
    assert "Capital Summary" in result.stdout

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert summary_csv.stat().st_size > 0

    trade_history_csv = reports_dir / "trade_history.csv"
    highlights_json = reports_dir / "highlights.json"
    assert trade_history_csv.is_file()
    assert trade_history_csv.stat().st_size > 0
    rows = list(csv.DictReader(trade_history_csv.open()))
    assert rows, "Trade history CSV empty"
    first = rows[0]
    assert first["action"] == "buy"
    prices = investor_demo.load_prices()
    assert float(first["price"]) == prices[0]
    assert highlights_json.is_file()
    assert highlights_json.stat().st_size > 0

    shutil.rmtree(reports_dir)


@pytest.mark.integration
def test_investor_demo_cli(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    # First run with packaged data
    out1 = tmp_path / "run1"
    cmd1 = [sys.executable, "scripts/investor_demo.py", "--reports", str(out1)]
    _run_and_check(cmd1, out1, repo_root, env)

    # Second run using explicit data file
    out2 = tmp_path / "run2"
    data_path = repo_root / "tests" / "data" / "prices.json"
    cmd2 = [
        sys.executable,
        "scripts/investor_demo.py",
        "--data",
        str(data_path),
        "--reports",
        str(out2),
    ]
    _run_and_check(cmd2, out2, repo_root, env)
