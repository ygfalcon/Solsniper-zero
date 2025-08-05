import json
import sys
import types

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
