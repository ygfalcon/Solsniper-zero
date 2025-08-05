import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_investor_demo_cli(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "scripts/investor_demo.py", "--reports", str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Wrote reports" in result.stdout
    assert "Capital Summary" in result.stdout
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert summary_csv.stat().st_size > 0
    trade_history_csv = tmp_path / "trade_history.csv"
    highlights_json = tmp_path / "highlights.json"
    assert trade_history_csv.is_file()
    assert trade_history_csv.stat().st_size > 0
    assert highlights_json.is_file()
    assert highlights_json.stat().st_size > 0
