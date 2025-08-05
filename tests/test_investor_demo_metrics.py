import json
import sys
import types

from solhunter_zero import investor_demo
import solhunter_zero.resource_monitor as rm


def test_resource_metrics_in_highlights(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 11.0)
    psutil_stub = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(percent=22.0)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 11.0
    assert highlights.get("memory_percent") == 22.0
