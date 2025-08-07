import subprocess
import sys
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_lines(name: str) -> list[str]:
    return (REPO_ROOT / name).read_text().splitlines()


def test_run_sh_invokes_startup():
    lines = _script_lines("run.sh")
    assert any("python -m solhunter_zero.startup" in line for line in lines)


def test_start_command_invokes_startup():
    lines = _script_lines("start.command")
    assert any("python -m solhunter_zero.startup" in line for line in lines)


def test_start_py_invokes_startup(tmp_path):
    start_src = REPO_ROOT / "start.py"
    tmp_start = tmp_path / "start.py"
    tmp_start.write_text(start_src.read_text())

    called = tmp_path / "called.txt"
    pkg = tmp_path / "solhunter_zero"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    stub = pkg / "startup.py"
    stub.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )

    subprocess.run([sys.executable, str(tmp_start), "EXTRA"], check=True, cwd=tmp_path)

    assert called.read_text().split() == ["EXTRA"]
