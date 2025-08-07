import logging
import runpy
import sys
from pathlib import Path

import pytest


def test_demo_handles_missing_optional_dependencies(monkeypatch, caplog):
    for mod in ["sklearn", "faiss", "scipy"]:
        monkeypatch.setitem(sys.modules, mod, None)
    caplog.set_level(logging.WARNING)
    script = Path(__file__).resolve().parents[1] / "one_click_trading_demo.py"
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(script, run_name="__main__")
    assert exc.value.code == 0
    text = caplog.text
    assert "StrangeAttractorAgent unavailable" in text
