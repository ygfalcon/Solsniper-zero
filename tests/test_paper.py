"""Test the paper trading CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from solhunter_zero.util import run_coro
from tests import stubs


def test_paper_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    reports = tmp_path / "reports"
    stubs.stub_sqlalchemy()
    import paper

    class TrackingMemory(paper.SyncMemory):
        def __init__(self) -> None:  # pragma: no cover - trivial
            super().__init__()
            self.logged: list[dict] = []

        def log_trade(self, **kwargs):  # type: ignore[override]
            self.logged.append(dict(kwargs))
            return super().log_trade(**kwargs)

    mem = TrackingMemory()
    monkeypatch.setattr(paper, "SyncMemory", lambda: mem)

    paper.run(["--reports", str(reports)])
    out = capsys.readouterr().out
    assert "ROI by agent" in out
    data = json.loads((reports / "paper_roi.json").read_text())
    # Buy-and-hold strategy should generate positive ROI on sample data
    assert data["buy_hold"] > 0
    # Trades were logged for each strategy
    trades = run_coro(mem.list_trades(limit=1000))
    assert len(trades) >= 18
