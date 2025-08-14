import importlib
import sys
import types
from pathlib import Path
import logging
import pytest


def stub_module(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    sys.modules[name] = mod


def test_monitor_processes_no_child_processes(monkeypatch, caplog):
    root = Path(__file__).resolve().parents[1]

    stub_module(
        "solhunter_zero.bootstrap_utils",
        ensure_venv=lambda argv=None: None,
        prepend_repo_root=lambda: None,
        ensure_cargo=lambda: None,
    )
    stub_module("solhunter_zero.logging_utils", log_startup=lambda msg: None)
    stub_module("solhunter_zero.paths", ROOT=root)
    stub_module(
        "solhunter_zero.device",
        ensure_gpu_env=lambda: None,
        detect_gpu=lambda: False,
        get_gpu_backend=lambda: "cpu",
        get_default_device=lambda: "cpu",
    )
    stub_module(
        "solhunter_zero.system",
        set_rayon_threads=lambda: None,
        detect_cpu_count=lambda: 1,
    )
    stub_module("solhunter_zero.data_sync", stop_scheduler=lambda: None)
    stub_module(
        "solhunter_zero.autopilot",
        _maybe_start_event_bus=lambda cfg: None,
        shutdown_event_bus=lambda: None,
    )
    stub_module(
        "solhunter_zero.bootstrap",
        bootstrap=lambda one_click=True: None,
        ensure_keypair=lambda: None,
    )

    start_all = importlib.import_module("scripts.start_all")

    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as excinfo:
        with start_all.ProcessManager() as pm:
            pm.monitor_processes()
    assert excinfo.value.code != 0
    assert "No child processes were started" in caplog.text
