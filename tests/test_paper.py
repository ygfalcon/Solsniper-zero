"""Test the paper trading CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from tests import stubs


class _Analyzer:
    """Minimal ``TradeAnalyzer`` replacement working on ``SimpleMemory``."""

    def __init__(self, memory) -> None:  # pragma: no cover - trivial
        self.memory = memory

    def roi_by_agent(self) -> dict[str, float]:  # pragma: no cover - trivial
        trades = self.memory.list_trades(limit=1000)
        summary: dict[str, dict[str, float]] = {}
        for t in trades:
            name = str(t.get("reason") or "")
            info = summary.setdefault(name, {"buy": 0.0, "sell": 0.0})
            direction = str(t.get("direction"))
            info[direction] += float(t.get("amount", 0)) * float(t.get("price", 0))
        rois = {}
        for name, info in summary.items():
            spent = info.get("buy", 0.0)
            revenue = info.get("sell", 0.0)
            if spent > 0:
                rois[name] = (revenue - spent) / spent
        return rois


def test_paper_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    reports = tmp_path / "reports"
    stubs.stub_sqlalchemy()
    import paper

    monkeypatch.setattr(paper, "TradeAnalyzer", _Analyzer)
    paper.run(["--reports", str(reports)])
    out = capsys.readouterr().out
    assert "ROI by agent" in out
    data = json.loads((reports / "paper_roi.json").read_text())
    # ROI should be positive for the synthetic trades
    assert next(iter(data.values())) > 0
