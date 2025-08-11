"""Test the paper trading CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from solhunter_zero.util import run_coro
from solhunter_zero.datasets.sample_ticks import load_sample_ticks
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
    monkeypatch.setattr(paper, "fetch_live_ticks", lambda: load_sample_ticks())

    paper.run(["--reports", str(reports)])
    out = capsys.readouterr().out
    assert "ROI by agent" in out
    data = json.loads((reports / "paper_roi.json").read_text())
    # ROI should be positive for the synthetic trades
    assert next(iter(data.values())) > 0
    # Trades were logged via the asynchronous memory path
    trades = run_coro(mem.list_trades(limit=1000))
    assert len(trades) == 2
