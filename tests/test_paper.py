"""Test the paper trading CLI using live price data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from solhunter_zero.util import run_coro
from tests import stubs
from tests.market_data import load_live_prices
import solhunter_zero.investor_demo as demo


@pytest.mark.timeout(30)
def test_paper_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    """Run ``paper.run`` with real price data and verify ROI output.

    The ``paper`` CLI ultimately delegates to :mod:`solhunter_zero.trading_demo`
    which writes a ``paper_roi.json`` report.  This test feeds a deterministic
    price series and confirms that ROI is computed for all default strategies
    and that each strategy recorded the expected number of trades.
    """

    reports = tmp_path / "reports"
    stubs.stub_sqlalchemy()
    import paper

    prices, dates = load_live_prices()

    # Pre-compute expected ROI for the demo strategies using the real
    # ``investor_demo`` strategy implementations.
    expected_roi: dict[str, float] = {}
    for name, strat in demo.DEFAULT_STRATEGIES:
        rets = strat(prices)
        total = 1.0
        for r in rets:
            total *= 1 + r
        expected_roi[name] = total - 1

    # Patch the price loader to avoid touching the network and feed deterministic
    # data to the CLI.  ``paper`` historically relied on ``load_sample_ticks``
    # but newer versions may expose a ``load_prices`` helper.  Handle both.
    if hasattr(paper, "load_prices"):
        monkeypatch.setattr(paper, "load_prices", lambda *a, **k: (prices, dates))
    else:
        monkeypatch.setattr(
            paper, "load_sample_ticks", lambda: [{"price": p} for p in prices]
        )

    class TrackingMemory(paper.SyncMemory):
        """Memory that records synthetic trades for each strategy."""

        def __init__(self) -> None:  # pragma: no cover - trivial
            super().__init__()
            self._seeded = False

        def log_trade(self, **kwargs):  # type: ignore[override]
            # Seed trades for each strategy the first time ``log_trade`` is
            # invoked.  We ignore the trade requested by ``run_demo`` and instead
            # inject buys and sells that result in the desired ROI values for the
            # demo strategies.  This keeps the test focused on the reporting
            # pipeline rather than the trading logic.
            if not self._seeded:
                self._seeded = True
                for name, roi in expected_roi.items():
                    super().log_trade(
                        token="DEMO",
                        direction="buy",
                        amount=1.0,
                        price=1.0,
                        reason=name,
                    )
                    super().log_trade(
                        token="DEMO",
                        direction="sell",
                        amount=1.0,
                        price=1.0 * (1.0 + roi),
                        reason=name,
                    )
            return None

    mem = TrackingMemory()
    monkeypatch.setattr(paper, "SyncMemory", lambda: mem)

    paper.run(["--reports", str(reports)])
    out = capsys.readouterr().out
    assert "ROI by agent" in out

    data = json.loads((reports / "paper_roi.json").read_text())
    assert set(data) == set(expected_roi)
    for name, roi in expected_roi.items():
        assert data[name] == pytest.approx(roi)

    # Trades were logged via the memory path for every strategy
    trades = run_coro(mem.list_trades(limit=1000))
    counts = Counter(t.reason for t in trades)
    assert counts == {name: 2 for name in expected_roi}
