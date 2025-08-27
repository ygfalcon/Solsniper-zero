import importlib
import json
import sys
import types
from unittest.mock import patch

import pytest


def test_start_all_respects_ui_selftest_exit_code(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    monkeypatch.setenv("SELFTEST_SKIP_ARTIFACTS", "1")
    monkeypatch.setenv("CI", "true")

    def stub_module(name: str, **attrs) -> None:
        mod = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(mod, attr, value)
        monkeypatch.setitem(sys.modules, name, mod)

    stub_module(
        "solhunter_zero.bootstrap_utils",
        ensure_venv=lambda argv=None: None,
        prepend_repo_root=lambda: None,
        ensure_cargo=lambda: None,
    )
    stub_module("solhunter_zero.logging_utils", log_startup=lambda msg: None)
    stub_module("solhunter_zero.device", ensure_gpu_env=lambda: None)
    stub_module("solhunter_zero.system", set_rayon_threads=lambda: None)
    stub_module(
        "solhunter_zero.config",
        set_env_from_config=lambda *a, **k: None,
        ensure_config_file=lambda: None,
        validate_env=lambda: None,
        initialize_event_bus=lambda: None,
        REQUIRED_ENV_VARS=[],
    )
    stub_module("solhunter_zero.data_sync")
    stub_module(
        "solhunter_zero.service_launcher",
        start_depth_service=lambda *a, **k: None,
        start_rl_daemon=lambda *a, **k: None,
        wait_for_depth_ws=lambda *a, **k: None,
    )
    stub_module(
        "solhunter_zero.autopilot",
        _maybe_start_event_bus=lambda cfg: None,
        shutdown_event_bus=lambda: None,
    )
    stub_module("solhunter_zero.bootstrap")
    stub_module("solhunter_zero.event_bus")
    stub_module("solhunter_zero.ui", ui_selftest=lambda: 0)

    start_all = importlib.import_module("scripts.start_all")

    with patch("scripts.start_all.ui_selftest", return_value=2):
        with pytest.raises(SystemExit) as exc:
            start_all.main()
        assert exc.value.code == 2


def test_ui_selftest_writes_diagnostics_on_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SELFTEST_SKIP_ARTIFACTS", "1")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("SOLHUNTER_CONFIG", "missing.toml")

    from solhunter_zero.ui import ui_selftest

    rc = ui_selftest()
    assert rc != 0
    diag_file = tmp_path / "diagnostics.json"
    assert diag_file.exists()
    data = json.loads(diag_file.read_text(encoding="utf-8"))
    assert data.get("ok") is False
    assert "error" in data and isinstance(data["error"], str) and data["error"]

