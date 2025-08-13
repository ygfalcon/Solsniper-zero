import importlib.util
import threading
import types
import sys
from pathlib import Path

import pytest

from solhunter_zero.event_bus import publish, BUS


def stub_module(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    sys.modules[name] = mod


def load_start_all():
    root = Path(__file__).resolve().parents[1]
    stub_module(
        "solhunter_zero.bootstrap_utils",
        ensure_venv=lambda argv=None: None,
        prepend_repo_root=lambda: None,
        ensure_cargo=lambda *a, **k: None,
    )
    stub_module("solhunter_zero.logging_utils", log_startup=lambda msg: None)
    stub_module("solhunter_zero.paths", ROOT=root)
    stub_module("solhunter_zero.device", ensure_gpu_env=lambda: None)
    stub_module("solhunter_zero.system", set_rayon_threads=lambda: None)
    stub_module(
        "solhunter_zero.config",
        REQUIRED_ENV_VARS=[],
        set_env_from_config=lambda *a, **k: None,
        ensure_config_file=lambda *a, **k: None,
        validate_env=lambda *a, **k: {},
        initialize_event_bus=lambda: None,
        reload_active_config=lambda: None,
    )
    stub_module("solhunter_zero.data_sync", stop_scheduler=lambda: None)
    stub_module(
        "solhunter_zero.service_launcher",
        start_depth_service=lambda *a, **k: None,
        start_rl_daemon=lambda: None,
        wait_for_depth_ws=lambda *a, **k: None,
    )
    stub_module(
        "solhunter_zero.autopilot",
        _maybe_start_event_bus=lambda cfg: None,
        shutdown_event_bus=lambda: None,
    )
    stub_module(
        "solhunter_zero.ui",
        rl_ws_loop=None,
        event_ws_loop=None,
        log_ws_loop=None,
        start_websockets=lambda: {},
        create_app=lambda: None,
    )
    stub_module(
        "solhunter_zero.bootstrap",
        bootstrap=lambda *a, **k: None,
        ensure_keypair=lambda: None,
    )
    spec = importlib.util.spec_from_file_location(
        "start_all", root / "scripts/start_all.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


start_all = load_start_all()


class DummyProc:
    """Simple stand-in for ``subprocess.Popen`` used in tests."""

    returncode = None

    def poll(self):  # pragma: no cover - trivial
        return None


def setup_function(func):
    BUS.reset()


def test_wait_for_rl_daemon_receives_heartbeat():
    proc = DummyProc()
    t = threading.Timer(0.01, lambda: publish("heartbeat", {"service": "rl_daemon"}))
    t.start()
    start_all._wait_for_rl_daemon(proc, timeout=1.0)
    t.join()


def test_wait_for_rl_daemon_times_out():
    proc = DummyProc()
    with pytest.raises(TimeoutError):
        start_all._wait_for_rl_daemon(proc, timeout=0.05)

