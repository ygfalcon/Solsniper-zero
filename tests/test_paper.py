"""Test the paper trading CLI wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_paper_cli(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "paper.py", "--reports", str(reports)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ROI by agent" in result.stdout
    data = json.loads((reports / "paper_roi.json").read_text())
    # ROI should be positive for the synthetic trades
    assert next(iter(data.values())) > 0
