import csv
import importlib.util
import json
from pathlib import Path


def test_investor_demo(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "investor_demo", Path("scripts/investor_demo.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.main(
        [
            "--data",
            "tests/data/prices.json",
            "--reports",
            str(tmp_path),
            "--capital",
            "100",
        ]
    )

    summary_json = tmp_path / "summary.json"
    assert summary_json.exists(), "Summary JSON not generated"
    content = json.loads(summary_json.read_text())
    assert content and "roi" in content[0]

    summary_csv = tmp_path / "summary.csv"
    assert summary_csv.exists(), "Summary CSV not generated"
    with summary_csv.open(newline="", encoding="utf-8") as cf:
        reader = csv.reader(cf)
        header = next(reader)
    assert {"roi", "sharpe", "drawdown"}.issubset(header)

