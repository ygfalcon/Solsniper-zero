import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, tmp_path: Path, monkeypatch) -> list[str]:
    """Run *script* and capture the args passed to the Python interpreter.

    A temporary ``python`` executable is placed at the front of ``PATH`` which
    writes the received arguments to a file. This allows us to assert which
    script and options the shell wrappers forward without executing the real
    launcher.
    """

    called = tmp_path / "called.txt"
    stub = tmp_path / "python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import sys, pathlib\n"
        f"pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    subprocess.run([str(REPO_ROOT / script), "EXTRA"], check=True, cwd=REPO_ROOT)

    return called.read_text().split()


def test_start_command_invokes_launcher(monkeypatch, tmp_path):
    if platform.machine() != "arm64":
        pytest.skip("start.command requires arm64")
    args = _run_script("start.command", tmp_path, monkeypatch)
    assert args[:3] == ["start.py", "--one-click", "--full-deps"]
    assert args[-1] == "EXTRA"


def test_run_sh_invokes_launcher(monkeypatch, tmp_path):
    args = _run_script("run.sh", tmp_path, monkeypatch)
    assert args[:3] == ["start.py", "--one-click", "--full-deps"]
    assert args[-1] == "EXTRA"

