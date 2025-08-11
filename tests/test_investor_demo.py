"""Test the investor demo CLI wrapper."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path: Path) -> None:
    """Run the demo via its CLI entry point and inspect key outputs."""

    reports = tmp_path / "reports"
    repo = Path(__file__).resolve().parents[1]
    cmd = ["./demo.command", "--preset", "short", "--reports", str(reports)]
    env = os.environ | {
        "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "SOLHUNTER_TESTING": "1",
        "SOLHUNTER_PATCH_INVESTOR_DEMO": "1",
    }
    proc = subprocess.run(
        cmd, cwd=repo, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr

    captured = proc.stdout
    assert "Capital Summary:" in captured

    match = re.search(r"Trade type results: (\{.*\})", captured)
    assert match, "trade results missing from output"
    results = json.loads(match.group(1))
    assert results["flash_loan_signature"] == "sig"
    assert results["arbitrage_path"] == ["dex1", "dex2"]

    highlights = json.loads((reports / "highlights.json").read_text())
    assert "top_strategy" in highlights

    summary = json.loads((reports / "summary.json").read_text())
    assert any(item.get("trades", 0) for item in summary)
