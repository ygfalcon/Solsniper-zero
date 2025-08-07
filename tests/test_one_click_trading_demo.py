import json
import os
import subprocess
import sys
from pathlib import Path

from solhunter_zero.investor_demo import discover_agent_strategies


def test_one_click_trading_demo(tmp_path):
    reports = tmp_path / "reports"
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    cmd = [sys.executable, "scripts/one_click_trading_demo.py", "--reports", str(reports)]
    subprocess.run(cmd, check=True, env=env)

    _, strategies = discover_agent_strategies()
    names = [name for name, _ in strategies]
    for name in names:
        assert (reports / f"{name}.json").is_file()

    portfolio_json = reports / "portfolio.json"
    portfolio_csv = reports / "portfolio.csv"
    assert portfolio_json.is_file()
    assert portfolio_csv.is_file()
    data = json.loads(portfolio_json.read_text())
    assert "roi" in data
