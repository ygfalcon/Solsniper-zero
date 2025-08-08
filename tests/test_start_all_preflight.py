import importlib
import sys
import types

import pytest


def _stub_modules(monkeypatch, *, start_scheduler=None):
    if start_scheduler is None:
        start_scheduler = lambda **k: None

    dummy_env = types.SimpleNamespace(load_env_file=lambda path: None)
    monkeypatch.setitem(sys.modules, "solhunter_zero.env", dummy_env)

    dummy_device = types.SimpleNamespace(ensure_gpu_env=lambda: None)
    monkeypatch.setitem(sys.modules, "solhunter_zero.device", dummy_device)

    dummy_system = types.SimpleNamespace(set_rayon_threads=lambda: None)
    monkeypatch.setitem(sys.modules, "solhunter_zero.system", dummy_system)

    dummy_config = types.SimpleNamespace(
        set_env_from_config=lambda cfg: None,
        ensure_config_file=lambda: None,
        validate_env=lambda env_vars, cfg: {},
        REQUIRED_ENV_VARS=(),
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", dummy_config)

    dummy_data_sync = types.SimpleNamespace(
        start_scheduler=start_scheduler, stop_scheduler=lambda: None
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.data_sync", dummy_data_sync)

    dummy_service_launcher = types.SimpleNamespace(
        start_depth_service=lambda cfg, stream_stderr=False: None,
        start_rl_daemon=lambda: None,
        wait_for_depth_ws=lambda *a, **k: None,
    )
    monkeypatch.setitem(
        sys.modules, "solhunter_zero.service_launcher", dummy_service_launcher
    )

    dummy_bootstrap = types.SimpleNamespace(ensure_cargo=lambda: None)
    monkeypatch.setitem(sys.modules, "solhunter_zero.bootstrap_utils", dummy_bootstrap)

    dummy_logging = types.SimpleNamespace(
        log_startup=lambda *a, **k: None, setup_logging=lambda *a, **k: None
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.logging_utils", dummy_logging)


def test_preflight_failure_aborts(monkeypatch, capsys):
    dummy_preflight = types.SimpleNamespace(
        run_preflight=lambda: [("Test", False, "boom")]
    )
    monkeypatch.setitem(sys.modules, "scripts.preflight", dummy_preflight)

    _stub_modules(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["start_all.py"])
    sys.modules.pop("scripts.start_all", None)
    with pytest.raises(SystemExit) as exc:
        importlib.import_module("scripts.start_all")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Test: FAIL - boom" in out


def test_skip_preflight(monkeypatch):
    called = {"ran": False}

    def fake_run_preflight():
        called["ran"] = True
        return []

    dummy_preflight = types.SimpleNamespace(run_preflight=fake_run_preflight)
    monkeypatch.setitem(sys.modules, "scripts.preflight", dummy_preflight)

    def stop_scheduler(**kwargs):
        raise SystemExit(0)

    _stub_modules(monkeypatch, start_scheduler=stop_scheduler)

    monkeypatch.setattr(sys, "argv", ["start_all.py", "--skip-preflight"])
    sys.modules.pop("scripts.start_all", None)
    with pytest.raises(SystemExit) as exc:
        importlib.import_module("scripts.start_all")
    assert exc.value.code == 0
    assert not called["ran"]
