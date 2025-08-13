import os
import sys
import types
import importlib
import subprocess
import shutil
from pathlib import Path


def test_event_bus_env_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    monkeypatch.delenv("BROKER_WS_URLS", raising=False)
    monkeypatch.delenv("EVENT_BUS_URL", raising=False)

    # Stub proto and config dependencies used by event_bus
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.event_pb2",
        types.ModuleType("solhunter_zero.event_pb2"),
    )
    config_mod = types.ModuleType("solhunter_zero.config")
    config_mod.get_event_bus_peers = lambda cfg=None: []
    config_mod.get_event_bus_url = lambda cfg=None: os.getenv("EVENT_BUS_URL", "")
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", config_mod)

    event_bus = importlib.import_module("solhunter_zero.event_bus")
    DEFAULT_WS_URL = event_bus.DEFAULT_WS_URL

    # Minimal environment configuration stub that writes BROKER_WS_URLS to .env
    def fake_configure_environment(root: Path):
        env_file = Path(root) / ".env"
        env_file.write_text("")
        with env_file.open("a", encoding="utf-8") as fh:
            fh.write(f"BROKER_WS_URLS={DEFAULT_WS_URL}\n")
        os.environ["BROKER_WS_URLS"] = DEFAULT_WS_URL
        return {"BROKER_WS_URLS": DEFAULT_WS_URL}

    env_config_mod = types.ModuleType("solhunter_zero.env_config")
    env_config_mod.configure_environment = fake_configure_environment
    monkeypatch.setitem(sys.modules, "solhunter_zero.env_config", env_config_mod)

    # Stub remaining heavy dependencies
    mac_mod = types.ModuleType("solhunter_zero.macos_setup")
    mac_mod.ensure_tools = lambda non_interactive=True: None
    monkeypatch.setitem(sys.modules, "solhunter_zero.macos_setup", mac_mod)

    device_mod = types.ModuleType("solhunter_zero.device")
    device_mod.initialize_gpu = lambda: None
    device_mod.METAL_EXTRA_INDEX = []
    monkeypatch.setitem(sys.modules, "solhunter_zero.device", device_mod)

    wallet_mod = types.ModuleType("solhunter_zero.wallet")
    wallet_mod.setup_default_keypair = lambda: None
    monkeypatch.setitem(sys.modules, "solhunter_zero.wallet", wallet_mod)

    qs_mod = types.ModuleType("scripts.quick_setup")
    qs_mod.main = lambda argv: None
    qs_mod.CONFIG_PATH = None
    monkeypatch.setitem(sys.modules, "scripts.quick_setup", qs_mod)

    log_mod = types.ModuleType("solhunter_zero.logging_utils")
    log_mod.log_startup = lambda msg: None
    monkeypatch.setitem(sys.modules, "solhunter_zero.logging_utils", log_mod)

    # Isolate filesystem interactions
    paths_mod = types.ModuleType("solhunter_zero.paths")
    paths_mod.ROOT = tmp_path
    monkeypatch.setitem(sys.modules, "solhunter_zero.paths", paths_mod)

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)

    setup_one_click = importlib.import_module("scripts.setup_one_click")
    monkeypatch.setattr(setup_one_click, "ROOT", tmp_path, raising=False)

    setup_one_click.main(["--dry-run"])

    env_file = tmp_path / ".env"
    lines = env_file.read_text().splitlines()
    assert f"EVENT_BUS_URL={DEFAULT_WS_URL}" in lines
    assert f"BROKER_WS_URLS={DEFAULT_WS_URL}" in lines

    assert event_bus._resolve_ws_urls({}) == {DEFAULT_WS_URL}

