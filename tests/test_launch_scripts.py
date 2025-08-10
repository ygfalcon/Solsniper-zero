import json
import shutil
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


def test_start_py_repair_flag_passed(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    pkg = tmp_path / "solhunter_zero"
    pkg.mkdir()

    # Copy the real launcher to exercise find_python and argument handling
    launcher_src = repo_root / "solhunter_zero" / "launcher.py"
    shutil.copy(launcher_src, pkg / "launcher.py")

    # Minimal supporting modules required by launcher.py
    (pkg / "__init__.py").write_text("")
    (pkg / "paths.py").write_text(
        "from pathlib import Path\nROOT = Path(__file__).resolve().parent.parent\n"
    )
    (pkg / "macos_setup.py").write_text("def ensure_tools(non_interactive=True):\n    pass\n")
    (pkg / "bootstrap_utils.py").write_text("def ensure_venv(arg):\n    pass\n")
    (pkg / "logging_utils.py").write_text(
        "def setup_logging(name):\n    pass\n\n"
        "def log_startup(msg):\n    pass\n"
    )
    (pkg / "env_config.py").write_text("def configure_environment(root):\n    pass\n")
    (pkg / "device.py").write_text("def initialize_gpu():\n    pass\n")
    (pkg / "system.py").write_text("def set_rayon_threads():\n    pass\n")

    # Entry point and stubbed startup module
    shutil.copy(repo_root / "start.py", tmp_path / "start.py")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "__init__.py").write_text("")
    marker = tmp_path / "argv.txt"
    (scripts_dir / "startup.py").write_text(
        f"import sys, pathlib; pathlib.Path(r'{marker}').write_text(' '.join(sys.argv[1:]))"
    )

    subprocess.run([sys.executable, str(tmp_path / "start.py"), "--repair"], check=True)

    recorded = marker.read_text().split()
    assert "--repair" in recorded


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
