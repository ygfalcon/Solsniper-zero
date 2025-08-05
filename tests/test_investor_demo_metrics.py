import json
import sys
import types

import pytest
from solhunter_zero import investor_demo
import solhunter_zero.resource_monitor as rm


def _patch_metrics(monkeypatch):
    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 11.0)
    psutil_stub = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(percent=22.0)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)


def test_resource_metrics_in_highlights(tmp_path, monkeypatch):
    _patch_metrics(monkeypatch)

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 11.0
    assert highlights.get("memory_percent") == 22.0


def test_resource_metrics_stdout(tmp_path, monkeypatch, capsys):
    _patch_metrics(monkeypatch)

    investor_demo.main(["--reports", str(tmp_path)])

    captured = capsys.readouterr()
    assert "CPU: 11.00% Memory: 22.00%" in captured.out


def test_no_resource_metrics_when_psutil_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError):
        __import__("psutil")

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert "cpu_usage" not in highlights
    assert "memory_percent" not in highlights

    captured = capsys.readouterr()
    assert "CPU:" not in captured.out
    assert "Memory:" not in captured.out
