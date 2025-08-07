import os
import platform
import runpy
import sys
import types
from pathlib import Path

import pytest

# Stub out the startup module so launching avoids real network activity
fake_startup = types.ModuleType("scripts.startup")

def _fake_run(argv):
    print("SolHunter Zero launch complete – system ready.")
    return 0

fake_startup.run = _fake_run
sys.modules["scripts.startup"] = fake_startup

macos_setup_mod = types.ModuleType("solhunter_zero.macos_setup")
macos_setup_mod.ensure_tools = lambda non_interactive=True: None
sys.modules["solhunter_zero.macos_setup"] = macos_setup_mod

bootstrap_utils_mod = types.ModuleType("solhunter_zero.bootstrap_utils")
bootstrap_utils_mod.ensure_venv = lambda arg: None
bootstrap_utils_mod.ensure_deps = lambda install_optional=False: None
bootstrap_utils_mod.ensure_endpoints = lambda cfg: None
bootstrap_utils_mod.METAL_EXTRA_INDEX = []
sys.modules["solhunter_zero.bootstrap_utils"] = bootstrap_utils_mod

logging_utils_mod = types.ModuleType("solhunter_zero.logging_utils")
logging_utils_mod.log_startup = lambda msg: None
logging_utils_mod.rotate_startup_log = lambda: None
sys.modules["solhunter_zero.logging_utils"] = logging_utils_mod

env_config_mod = types.ModuleType("solhunter_zero.env_config")
env_config_mod.configure_environment = lambda root: {}
sys.modules["solhunter_zero.env_config"] = env_config_mod

device_mod = types.ModuleType("solhunter_zero.device")
device_mod.initialize_gpu = lambda: None
device_mod.METAL_EXTRA_INDEX = []
sys.modules["solhunter_zero.device"] = device_mod

system_mod = types.ModuleType("solhunter_zero.system")
system_mod.set_rayon_threads = lambda: None
sys.modules["solhunter_zero.system"] = system_mod


def load_launcher():
    globals_dict = {
        "__name__": "scripts.launcher",
        "__file__": str(Path("scripts/launcher.py")),
        "find_python": lambda: sys.executable,
    }
    module_dict = runpy.run_path("scripts/launcher.py", globals_dict)
    mod = types.ModuleType("scripts.launcher")
    mod.__dict__.update(module_dict)
    return mod


def test_one_click_launch(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    dummy_device = types.ModuleType("device")
    dummy_device.initialize_gpu = lambda: None
    sys.modules["solhunter_zero.device"] = dummy_device

    launcher = load_launcher()
    monkeypatch.setattr(launcher, "device", dummy_device)

    def fake_execvp(prog, argv):
        if argv[0] == "arch":
            args = argv[4:]
        else:
            args = argv[2:]
        code = fake_startup.run(args)
        raise SystemExit(code)

    monkeypatch.setattr(os, "execvp", fake_execvp)

    with pytest.raises(SystemExit) as exc:
        launcher.main(["--one-click", "--skip-deps", "--skip-preflight"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "SolHunter Zero launch complete – system ready." in out
