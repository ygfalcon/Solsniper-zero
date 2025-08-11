import importlib
import sys
from pathlib import Path

import pytest


def test_parse_launcher_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--repair", "--fast", "extra"])
    launcher = importlib.reload(importlib.import_module("solhunter_zero.launcher"))

    args, forward = launcher.parse_launcher_args()

    assert args.repair is True
    assert args.fast is True
    assert forward == ["extra"]
    assert launcher.FAST_MODE is True


def test_ensure_interpreter_reexec(monkeypatch):
    launcher = importlib.reload(importlib.import_module("solhunter_zero.launcher"))

    class Args:
        repair = False

    monkeypatch.setattr(launcher, "find_python", lambda repair=False: "pythonX")
    called = {}

    def fake_execv(prog, argv):
        called["prog"] = prog
        called["argv"] = argv
        raise RuntimeError

    monkeypatch.setattr(launcher.os, "execv", fake_execv)

    with pytest.raises(RuntimeError):
        launcher.ensure_interpreter(Args(), ["a", "b"])

    assert called["prog"] == "pythonX"
    assert called["argv"][0] == "pythonX"
    assert called["argv"][1] == str(Path(launcher.__file__).resolve())
    assert called["argv"][2:] == ["a", "b"]

