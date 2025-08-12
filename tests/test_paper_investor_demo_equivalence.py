from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import demo
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

    # Convert ticks to a price dataset so the demo and paper scripts share input
    price_path = paper._ticks_to_price_file(ticks)

    reports_demo = tmp_path / "demo_reports"
    reports_paper = tmp_path / "paper_reports"
    monkeypatch.setenv("SOLHUNTER_PATCH_INVESTOR_DEMO", "1")

    capital = 250.0
    demo.run([
        "--url",
        str(price_path),
        "--reports",
        str(reports_demo),
        "--capital",
        str(capital),
    ])
    paper.run([
        "--reports",
        str(reports_paper),
        "--ticks",
        str(data_path),
        "--capital",
        str(capital),
    ])
    captured = capsys.readouterr().out
    assert "Capital Summary:" in captured

    match = re.search(r"Trade type results: (\{.*\})", captured)
    assert match, "trade results missing from output"
    results = json.loads(match.group(1))
    assert isinstance(results.get("flash_loan_signature"), str)
    assert isinstance(results.get("arbitrage_path"), list)

    highlights = json.loads((reports_paper / "highlights.json").read_text())
    assert "top_strategy" in highlights

    summary_demo = json.loads((reports_demo / "summary.json").read_text())
    summary_paper = json.loads((reports_paper / "summary.json").read_text())
    trade_hist = json.loads((reports_paper / "trade_history.json").read_text())
    assert_demo_reports(summary_paper, trade_hist)

    expected = capital * (1 + summary_demo[0]["roi"])
    assert summary_demo[0]["final_capital"] == pytest.approx(expected)
    assert summary_paper[0]["final_capital"] == pytest.approx(expected)
    assert summary_demo[0]["final_capital"] == pytest.approx(
        summary_paper[0]["final_capital"]
    )
