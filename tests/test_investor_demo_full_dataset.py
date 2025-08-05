import json
from pathlib import Path

import pytest

from solhunter_zero import investor_demo


MAX_DATA_POINTS = 5000


@pytest.mark.timeout(30)
def test_investor_demo_full_dataset(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "solhunter_zero" / "data" / "investor_demo_prices.json"
    data = json.loads(data_path.read_text())
    if len(data) > MAX_DATA_POINTS:
        pytest.skip("dataset too large for resource constraints")

    reports_dir = tmp_path / "reports"
    argv = ["--data", str(data_path), "--reports", str(reports_dir)]
    investor_demo.main(argv)

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    trade_history_csv = reports_dir / "trade_history.csv"
    trade_history_json = reports_dir / "trade_history.json"

    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert trade_history_csv.is_file()
    assert trade_history_json.is_file()
    assert summary_json.stat().st_size > 0
    assert trade_history_csv.stat().st_size > 0
