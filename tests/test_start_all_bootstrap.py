import importlib
import sys
import types
from pathlib import Path


def _stub(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_start_all_runs_with_stubbed_environment(monkeypatch):
    calls = []

    pkg = types.ModuleType("solhunter_zero")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "solhunter_zero", pkg)

    _stub(
        monkeypatch,
        "solhunter_zero.bootstrap_utils",
        ensure_venv=lambda argv: calls.append("venv"),
        ensure_cargo=lambda: None,
    )
    _stub(monkeypatch, "solhunter_zero.paths", ROOT=Path("."))
    _stub(
        monkeypatch,
        "solhunter_zero.logging_utils",
        log_startup=lambda msg: calls.append("log"),
    )
    _stub(
        monkeypatch,
        "solhunter_zero.device",
        ensure_gpu_env=lambda: calls.append("gpu"),
    )
    _stub(
        monkeypatch,
        "solhunter_zero.system",
        set_rayon_threads=lambda: calls.append("threads"),
    )
    _stub(
        monkeypatch,
        "solhunter_zero.config",
        REQUIRED_ENV_VARS=[],
        set_env_from_config=lambda cfg: None,
        ensure_config_file=lambda: "cfg",
        validate_env=lambda req, cfg: {},
        initialize_event_bus=lambda: None,
        reload_active_config=lambda: None,
    )
    _stub(
        monkeypatch,
        "solhunter_zero.data_sync",
        start_scheduler=lambda **k: None,
        stop_scheduler=lambda: None,
    )

    class DummyProc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, *a, **k):
            pass

        def kill(self):
            pass

    _stub(
        monkeypatch,
        "solhunter_zero.service_launcher",
        start_depth_service=lambda cfg, stream_stderr=False: DummyProc(),
        start_rl_daemon=lambda: DummyProc(),
        wait_for_depth_ws=lambda addr, port, deadline, proc: None,
    )
    _stub(
        monkeypatch,
        "solhunter_zero.autopilot",
        _maybe_start_event_bus=lambda cfg: None,
        shutdown_event_bus=lambda: None,
    )

    class DummyApp:
        def run(self):
            pass

    _stub(
        monkeypatch,
        "solhunter_zero.ui",
        rl_ws_loop=None,
        event_ws_loop=None,
        log_ws_loop=None,
        create_app=lambda: DummyApp(),
        start_websockets=lambda: {},
    )
    _stub(
        monkeypatch,
        "solhunter_zero.bootstrap",
        bootstrap=lambda one_click=True: None,
        ensure_keypair=lambda: None,
    )

    start_all = importlib.import_module("scripts.start_all")
    assert calls[0] == "venv"

    monkeypatch.setattr(start_all, "launch_services", lambda pm: None)
    monkeypatch.setattr(start_all, "launch_ui", lambda pm: None)

    class DummyPM:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def monitor_processes(self):
            pass

    monkeypatch.setattr(start_all, "ProcessManager", DummyPM)

    start_all.main()
