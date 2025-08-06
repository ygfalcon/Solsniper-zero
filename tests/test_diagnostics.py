import builtins
import builtins
import shutil
import subprocess
import sys
import types

import pytest

from scripts import diagnostics, startup


def test_collect_no_torch(monkeypatch, tmp_path):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("torch", None)
    monkeypatch.setattr("solhunter_zero.device.get_gpu_backend", lambda: None)
    monkeypatch.chdir(tmp_path)

    info = diagnostics.collect()
    assert info["torch"] == "not installed"
    assert info["config"] == "missing"
    assert "python" in info
    assert "gpu_backend" in info


def test_collect_with_torch_and_keypair(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.chdir(tmp_path)

    dummy_torch = types.SimpleNamespace(__version__="1.0")
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return dummy_torch
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        "solhunter_zero.device.get_gpu_backend", lambda: "torch"
    )
    dummy_wallet = types.SimpleNamespace(
        list_keypairs=lambda: ["a"], get_active_keypair_name=lambda: "a"
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.wallet", dummy_wallet)
    monkeypatch.setattr(sys.modules["solhunter_zero"], "wallet", dummy_wallet, raising=False)

    def fake_check_output(cmd, text=True):
        if cmd[0] == "rustc":
            return "rustc 1.70.0"
        if cmd[0] == "cargo":
            return "cargo 1.70.0"
        raise ValueError

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    monkeypatch.setattr("shutil.which", lambda cmd: cmd)

    info = diagnostics.collect()
    assert info["torch"] == "1.0"
    assert info["gpu_backend"] == "torch"
    assert info["config"] == "present"
    assert info["keypair"] == "a"
    assert info["rustc"].startswith("rustc")


def test_startup_diagnostics_flag(capsys):
    code = startup.run(["--diagnostics"])
    out = capsys.readouterr().out.lower()
    assert code == 0
    assert "python" in out


def test_startup_runs_diagnostics_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        startup, "ensure_deps", lambda install_optional=False: (_ for _ in ()).throw(SystemExit(2))
    )
    code = startup.run([])
    out = capsys.readouterr().out.lower()
    assert code == 2
    assert "python" in out
