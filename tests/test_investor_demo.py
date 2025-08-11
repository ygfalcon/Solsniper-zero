"""Test the investor demo CLI wrapper."""

from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path

import pytest

from solhunter_zero.simple_memory import SimpleMemory
import solhunter_zero.investor_demo as demo


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path: Path, monkeypatch, capsys) -> None:
    """Run the demo via its CLI entry point and inspect key outputs."""

    # Use the lightweight in-memory implementation to avoid SQLAlchemy.
    class _Mem(SimpleMemory):
        def __init__(self, *a, **k):
            super().__init__()

    monkeypatch.setattr(demo, "Memory", _Mem)

    # Stub out optional trade demonstrations so the CLI executes without
    # heavyweight dependencies.
    async def _fake_arbitrage() -> dict:
        demo.used_trade_types.add("arbitrage")
        return {"path": ["dex1", "dex2"], "profit": 1.0}

    async def _fake_flash() -> str:
        demo.used_trade_types.add("flash_loan")
        return "sig"

    async def _fake_sniper() -> list[str]:
        demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def _fake_dex() -> list[str]:
        demo.used_trade_types.add("dex_scanner")
        return ["pool1"]

    async def _fake_route() -> dict:
        demo.used_trade_types.add("route_ffi")
        return {"path": ["r1", "r2"], "profit": 0.5}

    async def _fake_jito() -> int:
        demo.used_trade_types.add("jito_stream")
        return 0

    monkeypatch.setattr(demo, "_demo_arbitrage", _fake_arbitrage)
    monkeypatch.setattr(demo, "_demo_flash_loan", _fake_flash)
    monkeypatch.setattr(demo, "_demo_sniper", _fake_sniper)
    monkeypatch.setattr(demo, "_demo_dex_scanner", _fake_dex)
    monkeypatch.setattr(demo, "_demo_route_ffi", _fake_route)
    monkeypatch.setattr(demo, "_demo_jito_stream", _fake_jito)

    reports = tmp_path / "reports"
    sys.argv = ["demo.py", "--preset", "short", "--reports", str(reports)]

    # Execute the CLI script in-process so the patches above take effect.
    runpy.run_path(Path("demo.py"), run_name="__main__")

    captured = capsys.readouterr().out
    assert "Capital Summary:" in captured

    match = re.search(r"Trade type results: (\{.*\})", captured)
    assert match, "trade results missing from output"
    results = json.loads(match.group(1))
    assert results["flash_loan_signature"] == "sig"
    assert results["arbitrage_path"] == ["dex1", "dex2"]

    highlights = json.loads((reports / "highlights.json").read_text())
    assert "top_strategy" in highlights
    assert (reports / "summary.json").exists()
