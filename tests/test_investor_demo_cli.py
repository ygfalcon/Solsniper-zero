import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import csv

import pytest

from solhunter_zero import investor_demo


def _run_and_check(
    cmd: list[str],
    reports_dir: Path,
    repo_root: Path,
    env: dict[str, str],
    data_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Wrote reports" in result.stdout
    assert "Capital Summary" in result.stdout

    strategies = {"buy_hold", "momentum", "mean_reversion", "mixed"}
    pattern = re.compile(r"^(buy_hold|momentum|mean_reversion|mixed):\s*(\d+(?:\.\d+)?)$", re.MULTILINE)
    matches = {name: val for name, val in pattern.findall(result.stdout)}
    assert strategies <= matches.keys(), f"Missing strategies: {strategies - matches.keys()}"
    for val in matches.values():
        float(val)

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert summary_csv.stat().st_size > 0
    summary_rows = list(csv.DictReader(summary_csv.open()))
    assert any(r["config"] == "mean_reversion" for r in summary_rows)

    trade_history_csv = reports_dir / "trade_history.csv"
    highlights_json = reports_dir / "highlights.json"
    assert trade_history_csv.is_file()
    assert trade_history_csv.stat().st_size > 0
    rows = list(csv.DictReader(trade_history_csv.open()))
    assert rows, "Trade history CSV empty"
    first = rows[0]
    assert first["action"] == "buy"
    prices = investor_demo.load_prices(data_path)
    assert float(first["price"]) == prices[0]
    assert any(r["strategy"] == "mean_reversion" for r in rows)
    assert highlights_json.is_file()
    assert highlights_json.stat().st_size > 0

    shutil.rmtree(reports_dir, ignore_errors=True)
    return result


@pytest.mark.timeout(30)
@pytest.mark.integration
def test_investor_demo_cli(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent

    psutil_stub = tmp_path / "psutil.py"
    psutil_stub.write_text(
        "class Process:\n"
        "    def cpu_percent(self, interval=None):\n"
        "        return 0.0\n"
        "\n"
        "def cpu_percent(interval=None):\n"
        "    return 11.0\n"
        "\n"
        "class _VM:\n"
        "    percent = 22.0\n"
        "\n"
        "def virtual_memory():\n"
        "    return _VM()\n"
    )

    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{repo_root}",
    }

    out = tmp_path / "run"
    data_path = repo_root / "tests" / "data" / "prices_short.json"
    cmd = [
        sys.executable,
        "scripts/investor_demo.py",
        "--data",
        str(data_path),
        "--reports",
        str(out),
    ]
    result = _run_and_check(cmd, out, repo_root, env, data_path)
    assert "Resource usage - CPU: 11.00% Memory: 22.00%" in result.stdout
