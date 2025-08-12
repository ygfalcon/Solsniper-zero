from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import paper

from tests.market_data import load_live_prices
from tests.test_investor_demo import assert_demo_reports

pytestmark = pytest.mark.timeout(60)


def test_paper_investor_demo_equivalence(tmp_path: Path, monkeypatch, capsys) -> None:
    """Ensure paper CLI mirrors investor demo output structure."""

    prices, dates = load_live_prices()
    ticks = [
        {"timestamp": d, "price": p}
        for p, d in zip(prices, dates)
    ]
    data_path = tmp_path / "ticks.json"
    data_path.write_text(json.dumps(ticks))

    reports = tmp_path / "reports"
    monkeypatch.setenv("SOLHUNTER_PATCH_INVESTOR_DEMO", "1")

    paper.run(["--reports", str(reports), "--ticks", str(data_path)])
    captured = capsys.readouterr().out
    assert "Capital Summary:" in captured

    match = re.search(r"Trade type results: (\{.*\})", captured)
    assert match, "trade results missing from output"
    results = json.loads(match.group(1))
    assert isinstance(results.get("flash_loan_signature"), str)
    assert isinstance(results.get("arbitrage_path"), list)

    highlights = json.loads((reports / "highlights.json").read_text())
    assert "top_strategy" in highlights

    summary = json.loads((reports / "summary.json").read_text())
    trade_hist = json.loads((reports / "trade_history.json").read_text())
    assert_demo_reports(summary, trade_hist)
