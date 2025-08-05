import os
import subprocess
import sys
from pathlib import Path
import json
import re

import pytest


@pytest.mark.integration
def test_investor_demo_cli(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "scripts/investor_demo.py", "--reports", str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary_json = tmp_path / "summary.json"
    assert summary_json.is_file()

    content = json.loads(summary_json.read_text())
    lines = [line for line in result.stdout.splitlines() if line.startswith("Config ")]
    assert len(lines) == len(content)
    for line in lines:
        match = re.match(r"Config (\w+): start=([0-9.]+), end=([0-9.]+)", line)
        assert match, f"Unexpected output line: {line}"
        name, start, end = match.groups()
        assert float(start) == 100.0
        expected = next(item["final_capital"] for item in content if item["config"] == name)
        assert abs(float(end) - expected) < 1e-6
