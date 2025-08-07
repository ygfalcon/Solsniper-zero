import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_lines(name: str) -> list[str]:
    return (REPO_ROOT / name).read_text().splitlines()


def test_run_sh_invokes_launcher():
    lines = _script_lines("run.sh")
    assert any("scripts/launcher.py" in line for line in lines)


def test_start_py_invokes_launcher(tmp_path):
    start_src = REPO_ROOT / "start.py"
    tmp_start = tmp_path / "start.py"
    tmp_start.write_text(start_src.read_text())

    called = tmp_path / "called.txt"
    stub = tmp_path / "scripts" / "launcher.py"
    stub.parent.mkdir()
    stub.write_text(
        "import sys, pathlib\n"
        f"def main(argv=None):\n    pathlib.Path(r'{called}').write_text(' '.join(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)

    subprocess.run([sys.executable, str(tmp_start), "EXTRA"], check=True)

    assert called.read_text().split() == ["EXTRA"]


def test_start_command_prefers_venv(tmp_path):
    src = REPO_ROOT / "start.command"
    start = tmp_path / "start.command"
    start.write_text(src.read_text())
    start.chmod(0o755)

    called = tmp_path / "called.txt"
    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo \"$@\" > \"{called}\"\n"
    )
    venv_py.chmod(0o755)

    subprocess.run([str(start)], check=True)

    assert called.read_text().split()[:2] == ["-m", "solhunter_zero.launcher"]


def test_start_command_falls_back_to_arch(tmp_path):
    src = REPO_ROOT / "start.command"
    start = tmp_path / "start.command"
    start.write_text(src.read_text())
    start.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    called = tmp_path / "called.txt"
    py = bin_dir / "python3"
    py.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo \"$@\" > \"{called}\"\n"
    )
    py.chmod(0o755)

    arch_args = tmp_path / "arch_arg.txt"
    arch = bin_dir / "arch"
    arch.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo \"$1\" > \"{arch_args}\"\n"
        "shift\n"
        "exec \"$@\"\n"
    )
    arch.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:" + env.get("PATH", "")

    subprocess.run([str(start)], check=True, env=env)

    assert arch_args.read_text().strip() == "-arm64"
    assert called.read_text().split()[:2] == ["-m", "solhunter_zero.launcher"]
