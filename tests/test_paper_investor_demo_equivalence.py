from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import paper
from solhunter_zero.trade_analyzer import TradeAnalyzer

from tests.market_data import load_live_prices
from tests.test_investor_demo import assert_demo_reports
from tests import stubs

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
    stubs.install_stubs()
    import solhunter_zero.investor_demo as demo

    async def _fake_flash() -> str:
        demo.used_trade_types.add("flash_loan")
        return "sig"

    async def _fake_dex() -> list[str]:
        demo.used_trade_types.add("dex_scanner")
        return ["pool1"]

    demo._demo_flash_loan = _fake_flash
    demo._demo_dex_scanner = _fake_dex

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

    metrics_paper = TradeAnalyzer.performance_from_history(trade_hist)
    for row in summary:
        strat = row["config"]
        if strat in metrics_paper:
            assert metrics_paper[strat]["roi"] == pytest.approx(row["roi"])
            assert metrics_paper[strat]["max_drawdown"] == pytest.approx(
                row["drawdown"]
            )

    demo_reports = tmp_path / "demo_reports"
    demo_data = [{"date": t["timestamp"], "price": t["price"]} for t in ticks]
    demo_data_path = tmp_path / "demo_data.json"
    demo_data_path.write_text(json.dumps(demo_data))
    demo.main(["--reports", str(demo_reports), "--data", str(demo_data_path)])
    demo_hist = json.loads((demo_reports / "trade_history.json").read_text())
    metrics_demo = TradeAnalyzer.performance_from_history(demo_hist)
    for strat, vals in metrics_paper.items():
        assert strat in metrics_demo
        assert metrics_demo[strat]["roi"] == pytest.approx(vals["roi"])
        assert metrics_demo[strat]["max_drawdown"] == pytest.approx(
            vals["max_drawdown"]
        )
