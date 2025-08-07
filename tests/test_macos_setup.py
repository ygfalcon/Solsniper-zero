import platform
import types

import solhunter_zero.macos_setup as ms


def test_ensure_tools_skips_when_marker_exists(monkeypatch, tmp_path):
    marker = tmp_path / ".cache" / "macos_tools_ok"
    marker.parent.mkdir(parents=True)
    marker.write_text("ok")
    monkeypatch.setattr(ms, "TOOLS_OK_MARKER", marker)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.setattr(ms.shutil, "which", lambda cmd: "/usr/bin/fake")

    class DummyCompleted:
        def __init__(self, returncode=0):
            self.returncode = returncode

    monkeypatch.setattr(ms.subprocess, "run", lambda *a, **k: DummyCompleted(0))

    called = {"value": False}

    def fake_prepare(**kwargs):
        called["value"] = True
        return {"steps": {}, "success": True}

    monkeypatch.setattr(ms, "prepare_macos_env", fake_prepare)

    report = ms.ensure_tools()
    assert report == {"steps": {}, "success": True, "missing": []}
    assert not called["value"]


def test_ensure_tools_runs_setup_and_marks(monkeypatch, tmp_path):
    marker = tmp_path / ".cache" / "macos_tools_ok"
    monkeypatch.setattr(ms, "TOOLS_OK_MARKER", marker)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.setattr(ms.shutil, "which", lambda cmd: "/usr/bin/fake")

    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        rc = 1 if calls["count"] == 1 else 0
        return types.SimpleNamespace(returncode=rc)

    monkeypatch.setattr(ms.subprocess, "run", fake_run)

    def fake_prepare(**kwargs):
        return {"steps": {}, "success": True}

    monkeypatch.setattr(ms, "prepare_macos_env", fake_prepare)

    report = ms.ensure_tools()
    assert calls["count"] >= 2
    assert report["success"] is True
    assert report["missing"] == []
    assert marker.exists()
