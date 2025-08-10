import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_lines(name: str) -> list[str]:
    return (REPO_ROOT / name).read_text().splitlines()


def test_start_command_invokes_launcher():
    lines = _script_lines("start.command")
    assert any("start.py" in line for line in lines)


def test_start_py_invokes_launcher(tmp_path):
    start_src = REPO_ROOT / "start.py"
    tmp_start = tmp_path / "start.py"
    tmp_start.write_text(start_src.read_text())

    called = tmp_path / "called.txt"
    stub = tmp_path / "solhunter_zero" / "launcher.py"
    stub.parent.mkdir()
    stub.write_text(
        "import sys, pathlib\n"
        f"def main(argv=None):\n    pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)

    subprocess.run([sys.executable, str(tmp_start), "EXTRA"], check=True)

    assert called.read_text().split() == ["EXTRA"]


def test_start_command_executes_launcher(tmp_path):
    cmd_src = REPO_ROOT / "start.command"
    py_src = REPO_ROOT / "start.py"

    tmp_cmd = tmp_path / "start.command"
    tmp_py = tmp_path / "start.py"
    tmp_cmd.write_text(cmd_src.read_text())
    tmp_py.write_text(py_src.read_text())
    tmp_cmd.chmod(0o755)
    tmp_py.chmod(0o755)

    called = tmp_path / "called.txt"
    stub = tmp_path / "solhunter_zero" / "launcher.py"
    stub.parent.mkdir()
    stub.write_text(
        "import sys, pathlib\n"
        f"def main(argv=None):\n    pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    proc = subprocess.run([str(tmp_cmd), "EXTRA"], cwd=tmp_path, env=env)

    assert proc.returncode == 0
    assert called.read_text().split() == ["EXTRA"]


@pytest.mark.timeout(60)
def test_demo_script_generates_reports(tmp_path: Path) -> None:
    """demo.py runs end-to-end and produces report artifacts."""
    snippet = (
        "import runpy, sys, pathlib;"
        f"repo=pathlib.Path(r'{REPO_ROOT}');"
        "sys.path.insert(0, str(repo));"
        "import tests.stubs as s; s.stub_torch();"
        "path=repo / 'demo.py';"
        "sys.argv=[str(path)];"
        "runpy.run_path(str(path), run_name='__main__')"
    )

    subprocess.run([sys.executable, "-c", snippet], cwd=tmp_path, check=True)

    reports = tmp_path / "reports"
    highlights = json.loads((reports / "highlights.json").read_text())
    for key in [
        "arbitrage_path",
        "flash_loan_signature",
        "sniper_tokens",
        "dex_new_pools",
    ]:
        assert key in highlights

    summary = json.loads((reports / "summary.json").read_text())
    assert summary
