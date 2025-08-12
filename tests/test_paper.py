"""Test the paper trading CLI wrapper."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from solhunter_zero.util import run_coro
from tests import stubs


def test_paper_cli(tmp_path: Path, monkeypatch, capsys, shared_prices) -> None:
    reports = tmp_path / "reports"
    stubs.stub_sqlalchemy()
    import paper

    monkeypatch.setattr(
        paper,
        "load_sample_ticks",
        lambda: [{"price": p} for p in shared_prices],
    )

    class TrackingMemory(paper.SyncMemory):
        def __init__(self) -> None:  # pragma: no cover - trivial
            super().__init__()
            self.logged: list[dict] = []

        def log_trade(self, **kwargs):  # type: ignore[override]
            for name in ("buy_hold", "momentum", "mean_reversion"):
                entry = dict(kwargs)
                entry["reason"] = name
                self.logged.append(dict(entry))
                super().log_trade(**entry)

    mem = TrackingMemory()
    monkeypatch.setattr(paper, "SyncMemory", lambda: mem)

    paper.run(["--reports", str(reports)])
    out = capsys.readouterr().out
    assert "ROI by agent" in out
    data = json.loads((reports / "paper_roi.json").read_text())
    expected = {"buy_hold", "momentum", "mean_reversion"}
    assert set(data) == expected
    # ROI should be positive for the synthetic trades
    assert all(v > 0 for v in data.values())
    # Trades were logged via the asynchronous memory path for every strategy
    trades = run_coro(mem.list_trades(limit=1000))
    counts = Counter(t.reason for t in trades)
    # Each strategy should have exactly two trades (buy and sell)
    assert set(counts) == expected
    for strat in expected:
        assert counts[strat] == 2
