import platform
import subprocess

import pytest

from solhunter_zero import bootstrap_utils


def test_ensure_deps_runs_prepare_macos_env(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    called = {"prepare": 0}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        return {"success": True}

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1


class FakeResult:
    def __init__(self, returncode=1, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_pip_install_logs_and_prints_summary(monkeypatch, capsys):
    messages: list[str] = []
    monkeypatch.setattr(bootstrap_utils, "log_startup", lambda m: messages.append(m))

    def fake_run(cmd, capture_output, text):
        return FakeResult(1, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        bootstrap_utils._pip_install("pkg", retries=1)

    out = capsys.readouterr().out
    assert "manual installation instructions" in out
    assert any("out" in m for m in messages)
    assert any("err" in m for m in messages)
