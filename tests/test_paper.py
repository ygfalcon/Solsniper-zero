from __future__ import annotations

import json
from pathlib import Path

import pytest
import paper


def test_paper_generates_reports(tmp_path):
    """Running the paper CLI should emit summaries based on strategy returns."""

    ticks = [
        {"timestamp": "2023-01-01", "price": 1.0},
        {"timestamp": "2023-01-02", "price": 2.0},
        {"timestamp": "2023-01-03", "price": 1.0},
    ]
    data_path = tmp_path / "ticks.json"
    data_path.write_text(json.dumps(ticks))

    reports = tmp_path / "reports"
    paper.run(["--reports", str(reports), "--ticks", str(data_path)])

    summary_path = reports / "summary.json"
    trade_path = reports / "trade_history.json"
    assert summary_path.exists()
    assert trade_path.exists()

    summary = json.loads(summary_path.read_text())
    momentum = next(r for r in summary if r["strategy"] == "momentum")
    assert momentum["roi"] == pytest.approx(1.0)

    trades = json.loads(trade_path.read_text())
    momentum_actions = [
        t["action"] for t in trades if t["strategy"] == "momentum"
    ]
    assert momentum_actions == ["buy", "buy", "hold"]

