import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_sh_is_symlink_to_start_py():
    script = REPO_ROOT / "run.sh"
    assert script.is_symlink()
    assert os.readlink(script) == "start.py"


def test_start_command_is_symlink_to_start_py():
    script = REPO_ROOT / "start.command"
    assert script.is_symlink()
    assert os.readlink(script) == "start.py"


def test_start_py_invokes_launcher(tmp_path):
    start_src = REPO_ROOT / "start.py"
    tmp_start = tmp_path / "start.py"
    tmp_start.write_text(start_src.read_text())

    called = tmp_path / "called.txt"
    stub_launcher = tmp_path / "scripts" / "launcher.py"
    stub_launcher.parent.mkdir()
    stub_launcher.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )
    stub_launcher.chmod(0o755)

    subprocess.run([sys.executable, str(tmp_start), "EXTRA"], check=True)

    assert called.read_text().split() == ["EXTRA"]
