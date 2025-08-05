import csv
import shutil
from pathlib import Path

import pytest

from solhunter_zero import investor_demo


def _run_and_check(reports_dir: Path, data_path: Path, monkeypatch) -> None:
    class DummyMem:
        def __init__(self, *a, **k) -> None:
            pass

        def log_var(self, _v: float) -> None:
            pass

        async def close(self) -> None:  # pragma: no cover - simple stub
            pass

    def fake_hedge(weights, corrs):  # pragma: no cover - simple stub
        return weights

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    investor_demo.main(
        [
            "--reports",
            str(reports_dir),
            "--data",
            str(data_path),
            "--max-periods",
            "20",
        ]
    )

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()
    rows = list(csv.DictReader(summary_csv.open()))
    assert any(r["config"] == "mean_reversion" for r in rows)

    trade_history_csv = reports_dir / "trade_history.csv"
    highlights_json = reports_dir / "highlights.json"
    assert trade_history_csv.is_file()
    assert trade_history_csv.stat().st_size > 0
    prices = investor_demo.load_prices(data_path)
    first = next(csv.DictReader(trade_history_csv.open()))
    assert float(first["price"]) == prices[0]
    assert highlights_json.is_file()

    shutil.rmtree(reports_dir)


@pytest.mark.integration
def test_investor_demo_cli(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent

    # Run with packaged data
    out1 = tmp_path / "run1"
    data1 = repo_root / "solhunter_zero" / "data" / "investor_demo_prices_small.json"
    _run_and_check(out1, data1, monkeypatch)

    # Run using explicit data file
    out2 = tmp_path / "run2"
    data2 = repo_root / "tests" / "data" / "prices.json"
    _run_and_check(out2, data2, monkeypatch)

