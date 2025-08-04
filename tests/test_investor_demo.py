import json
import subprocess
import sys


def test_investor_demo(tmp_path):
    data = [
        {"date": f"2024-01-{i+1:02d}", "price": 100 + i}
        for i in range(60)
    ]
    data_path = tmp_path / "prices.json"
    data_path.write_text(json.dumps(data))

    reports_dir = tmp_path / "reports"
    cmd = [sys.executable, "scripts/investor_demo.py", "--data", str(data_path), "--reports", str(reports_dir)]
    subprocess.check_call(cmd)

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    assert summary_json.exists(), "JSON report not generated"
    assert summary_csv.exists(), "CSV report not generated"

    content = json.loads(summary_json.read_text())
    assert isinstance(content, list) and content
    first = content[0]
    for key in ["config", "roi", "sharpe", "drawdown"]:
        assert key in first
