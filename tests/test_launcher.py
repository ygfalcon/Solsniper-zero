import importlib
import pytest


def _import_launcher():
    # Reload to ensure a clean module for each test
    import scripts.launcher as launcher
    importlib.reload(launcher)
    return launcher


def test_missing_arch(monkeypatch, capsys):
    launcher = _import_launcher()
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher, "_cpu_count", lambda py: 1)
    monkeypatch.setattr(launcher.shutil, "which", lambda cmd: None)
    with pytest.raises(SystemExit) as excinfo:
        launcher.main([])
    assert excinfo.value.code != 0
    assert "arch" in capsys.readouterr().err.lower()


def test_exec_failure(monkeypatch, capsys):
    launcher = _import_launcher()
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher, "_cpu_count", lambda py: 1)
    monkeypatch.setattr(launcher.shutil, "which", lambda cmd: "/usr/bin/arch")

    def fake_execvp(cmd0, cmd):
        raise OSError("boom")

    monkeypatch.setattr(launcher.os, "execvp", fake_execvp)
    with pytest.raises(SystemExit) as excinfo:
        launcher.main([])
    assert excinfo.value.code != 0
    assert "Error executing" in capsys.readouterr().err
