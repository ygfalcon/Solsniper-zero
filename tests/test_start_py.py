import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_start_py(tmp_path, exit_code):
    script_src = REPO_ROOT / "start.py"
    script_dst = tmp_path / "start.py"
    shutil.copy(script_src, script_dst)
    script_dst.chmod(0o755)

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "__init__.py").write_text("")
    (scripts_dir / "launcher.py").write_text(
        """import os, sys
from pathlib import Path

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    Path(os.environ['PYTHON_LOG']).write_text(' '.join(argv))
    raise SystemExit(int(os.environ.get('PYTHON_EXIT_CODE', '0')))
"""
    )

    env = os.environ.copy()
    env["PYTHON_LOG"] = str(tmp_path / "py.log")
    env["PYTHON_EXIT_CODE"] = str(exit_code)

    proc = subprocess.run([str(script_dst), "--skip-preflight"], cwd=tmp_path, env=env)
    log = Path(env["PYTHON_LOG"]).read_text().strip()
    return proc.returncode, log


def test_invokes_launcher(tmp_path):
    code, log = _run_start_py(tmp_path, exit_code=0)
    assert code == 0
    assert log == "--skip-preflight"


def test_propagates_failure(tmp_path):
    code, log = _run_start_py(tmp_path, exit_code=3)
    assert code == 3
    assert log == "--skip-preflight"
