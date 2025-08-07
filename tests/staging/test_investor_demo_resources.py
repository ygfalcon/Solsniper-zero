import json
import types

import psutil

import solhunter_zero.resource_monitor as rm
from solhunter_zero import investor_demo


def test_highlights_include_stubbed_resource_metrics(tmp_path, monkeypatch):
    """Investor demo writes stubbed CPU and memory metrics to highlights.json."""
    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 12.5)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(percent=67.5)
    )

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 12.5
    assert highlights.get("memory_percent") == 67.5
