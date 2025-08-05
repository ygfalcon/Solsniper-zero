import json
import sys
import types

from solhunter_zero import investor_demo
import solhunter_zero.resource_monitor as rm


def test_highlights_include_patched_cpu_and_memory(tmp_path, monkeypatch):
    """Ensure patched CPU and memory percentages are recorded."""
    # Patch resource monitor and psutil to return fixed metrics
    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 33.0)
    psutil_stub = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(percent=44.0)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 33.0
    assert highlights.get("memory_percent") == 44.0
