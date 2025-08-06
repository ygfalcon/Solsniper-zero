import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_start_command(tmp_path, exit_code):
    # Copy start.command and provide stub rotate_logs
    script_src = REPO_ROOT / "start.command"
    script_dst = tmp_path / "start.command"
    shutil.copy(script_src, script_dst)
    script_dst.chmod(0o755)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "rotate_logs.sh").write_text("rotate_logs() { :; }\n")

    # Stub python3 interpreter
    stub_py = tmp_path / "python3"
    stub_py.write_text(
        """#!/usr/bin/env bash
if [ "$1" = "-V" ]; then
  echo 'Python 3.11.0'
  exit 0
elif [ "$1" = "-m" ] && [ "$2" = "solhunter_zero.system" ] && [ "$3" = "cpu-count" ]; then
  echo 6
  exit 0
elif [ "$1" = "-" ]; then
  exit 0
fi
echo "$@" > "$PYTHON_LOG"
exit "${PYTHON_EXIT_CODE:-0}"
"""
    )
    stub_py.chmod(0o755)
    os.symlink(stub_py, tmp_path / "python3.11")

    # Provide minimal config to skip copying
    (tmp_path / "config.toml").write_text("")

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["PYTHON_LOG"] = str(tmp_path / "py.log")
    env["PYTHON_EXIT_CODE"] = str(exit_code)

    proc = subprocess.run([str(script_dst), "--skip-preflight"], cwd=tmp_path, env=env)
    log = Path(env["PYTHON_LOG"]).read_text().strip()
    return proc.returncode, log


def test_invokes_startup(tmp_path):
    code, log = _run_start_command(tmp_path, exit_code=0)
    assert code == 0
    assert log == "scripts/startup.py --one-click --full-deps --skip-preflight"


def test_propagates_failure(tmp_path):
    code, log = _run_start_command(tmp_path, exit_code=3)
    assert code == 3
    assert log == "scripts/startup.py --one-click --full-deps --skip-preflight"
