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
    assert any("solhunter_zero.launcher" in line for line in lines)



def test_startup_non_interactive(monkeypatch, tmp_path):
    from scripts import startup

    called = {}

    def fake_run(cmd, *a, **k):
        called["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(startup.subprocess, "run", fake_run)
    monkeypatch.delenv("SOLHUNTER_CONFIG", raising=False)
    monkeypatch.delenv("KEYPAIR_PATH", raising=False)

    cfg = tmp_path / "cfg.toml"
    kp = tmp_path / "kp.json"
    cfg.write_text("")
    kp.write_text("")

    code = startup.main([
        "--non-interactive",
        "--config",
        str(cfg),
        "--keypair",
        str(kp),
        "EXTRA",
    ])

    assert code == 0
    assert called["cmd"][1] == "scripts/start_all.py"
    assert called["cmd"][2:] == ["EXTRA"]
    assert os.environ["SOLHUNTER_CONFIG"] == str(cfg)
    assert os.environ["KEYPAIR_PATH"] == str(kp)


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
