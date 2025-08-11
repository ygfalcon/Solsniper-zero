"""Test the investor demo CLI wrapper."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path: Path, monkeypatch, capsys, shared_prices) -> None:
    """Run the demo and verify strategy ROI and trade history."""

    reports = tmp_path / "reports"
    monkeypatch.setenv("SOLHUNTER_PATCH_INVESTOR_DEMO", "1")
    from tests import stubs

    stubs.install_stubs()
    import solhunter_zero.investor_demo as demo

    monkeypatch.setattr(
        demo,
        "load_prices",
        lambda path=None, preset=None: (shared_prices, [str(i) for i in range(len(shared_prices))]),
    )

    demo.main(["--preset", "short", "--reports", str(reports)])
    captured = capsys.readouterr().out
    assert "Capital Summary:" in captured

    match = re.search(r"Trade type results: (\{.*\})", captured)
    assert match, "trade results missing from output"
    results = json.loads(match.group(1))
    assert results["flash_loan_signature"] == "sig"
    assert results["arbitrage_path"] == ["dex1", "dex2"]

    highlights = json.loads((reports / "highlights.json").read_text())
    assert "top_strategy" in highlights

    summary = json.loads((reports / "summary.json").read_text())
    trade_hist = json.loads((reports / "trade_history.json").read_text())
    expected = {"buy_hold", "momentum", "mean_reversion"}
    for strat in expected:
        row = next((r for r in summary if r["config"] == strat), None)
        assert row is not None
        assert row["trades"] > 0
        assert "roi" in row
    recorded = {t["strategy"] for t in trade_hist}
    assert expected.issubset(recorded)
