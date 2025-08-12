from __future__ import annotations

import json
from pathlib import Path

import pytest
import paper
import solhunter_zero.reports as report_schema


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
    paper.run(["--reports", str(reports), "--ticks", str(data_path)])

    summary, trade_hist, highlights = report_schema.load_reports(reports)
    momentum = next(r for r in summary if r.config == "momentum")
    assert momentum.roi == pytest.approx(1.0)
    momentum_actions = [
        t.action for t in trade_hist if t.strategy == "momentum"
    ]
    assert momentum_actions == ["buy", "buy", "hold"]
    assert highlights.top_strategy

