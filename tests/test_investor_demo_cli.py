import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

if getattr(np, "_STUB", False):
    pytest.skip("numpy required for investor demo", allow_module_level=True)


@pytest.mark.integration
def test_investor_demo_cli(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "scripts/investor_demo.py", "--reports", str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "summary.json").is_file()
