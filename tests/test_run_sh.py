import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_run_sh(tmp_path, exit_code):
    script_src = REPO_ROOT / "run.sh"
    script_dst = tmp_path / "run.sh"
    shutil.copy(script_src, script_dst)
    script_dst.chmod(0o755)

    stub_py = tmp_path / "python"
    stub_py.write_text(
        """#!/usr/bin/env bash
echo \"$@\" > \"$PYTHON_LOG\"
exit \"${PYTHON_EXIT_CODE:-0}\"
""",
    )
    stub_py.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["PYTHON_LOG"] = str(tmp_path / "py.log")
    env["PYTHON_EXIT_CODE"] = str(exit_code)

    proc = subprocess.run([str(script_dst), "--dry-run"], cwd=tmp_path, env=env)
    log = Path(env["PYTHON_LOG"]).read_text().strip()
    return proc.returncode, log


def test_invokes_launcher(tmp_path):
    code, log = _run_run_sh(tmp_path, exit_code=0)
    assert code == 0
    assert log == "scripts/launcher.py --one-click --full-deps --dry-run"


def test_propagates_failure(tmp_path):
    code, log = _run_run_sh(tmp_path, exit_code=5)
    assert code == 5
    assert log == "scripts/launcher.py --one-click --full-deps --dry-run"

