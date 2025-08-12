from __future__ import annotations

import json
from pathlib import Path

import pytest
from solhunter_zero.trade_analyzer import TradeAnalyzer
from tests import stubs


def test_paper_generates_reports(tmp_path, monkeypatch):
    """Running the paper CLI should emit summaries based on strategy returns."""

    ticks = [
        {"timestamp": "2023-01-01", "price": 1.0},
        {"timestamp": "2023-01-02", "price": 2.0},
        {"timestamp": "2023-01-03", "price": 1.0},
    ]
    data_path = tmp_path / "ticks.json"
    data_path.write_text(json.dumps(ticks))

    reports = tmp_path / "reports"
    monkeypatch.setenv("SOLHUNTER_PATCH_INVESTOR_DEMO", "1")
    stubs.install_stubs()
    import paper
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

    summary_path = reports / "summary.json"
    trade_path = reports / "trade_history.json"
    highlight_path = reports / "highlights.json"
    assert summary_path.exists()
    assert trade_path.exists()
    assert highlight_path.exists()

    summary = json.loads(summary_path.read_text())
    momentum = next(r for r in summary if r["config"] == "momentum")
    assert momentum["roi"] == pytest.approx(1.0)

    trades = json.loads(trade_path.read_text())
    momentum_actions = [
        t["action"] for t in trades if t["strategy"] == "momentum"
    ]
    assert momentum_actions == ["buy", "buy", "hold"]

    metrics = TradeAnalyzer.performance_from_history(trades)
    assert metrics["momentum"]["roi"] == pytest.approx(1.0)
    assert metrics["momentum"]["max_drawdown"] == pytest.approx(0.0)

