import os
import platform
import runpy
import sys
import types
import subprocess
import shutil
from pathlib import Path

import pytest

# Stub modules to avoid heavy side effects during import
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
logging_utils_mod.setup_logging = lambda name: None
sys.modules["solhunter_zero.logging_utils"] = logging_utils_mod

env_config_mod = types.ModuleType("solhunter_zero.env_config")
env_config_mod.configure_environment = lambda root: {}
env_config_mod.configure_startup_env = lambda root: {}
sys.modules["solhunter_zero.env_config"] = env_config_mod

sys.modules.setdefault("tomli_w", types.ModuleType("tomli_w"))

system_mod = types.ModuleType("solhunter_zero.system")
system_mod.set_rayon_threads = lambda: None
sys.modules["solhunter_zero.system"] = system_mod

# Provide dummy implementations that emit messages when invoked so the test can
# assert that the script attempted each action.

def _fake_gpu_env():
    env = {
        "SOLHUNTER_GPU_AVAILABLE": "0",
        "SOLHUNTER_GPU_DEVICE": "cpu",
        "TORCH_DEVICE": "cpu",
    }
    for k, v in env.items():
        print(f"{k}={v}")
        os.environ[k] = v
    return env

# Dummy device module

dummy_device = types.ModuleType("solhunter_zero.device")
dummy_device.detect_gpu = lambda: True
dummy_device.get_default_device = lambda: "cpu"
dummy_device.ensure_gpu_env = _fake_gpu_env
dummy_device.initialize_gpu = _fake_gpu_env
dummy_device.METAL_EXTRA_INDEX = []
sys.modules["solhunter_zero.device"] = dummy_device
if "solhunter_zero" in sys.modules:
    setattr(sys.modules["solhunter_zero"], "device", dummy_device)

# Dummy config utils with automatic keypair selection

def _fake_select_keypair(auto=True):
    print("Selected keypair: default")
    return types.SimpleNamespace(name="default", mnemonic_path=None)

config_utils_mod = types.ModuleType("solhunter_zero.config_utils")
config_utils_mod.select_active_keypair = _fake_select_keypair
config_utils_mod.ensure_default_config = lambda *a, **k: None
sys.modules["solhunter_zero.config_utils"] = config_utils_mod

# Dummy bootstrap module to simulate service launches
bootstrap_mod = types.ModuleType("solhunter_zero.bootstrap")
bootstrap_mod.ensure_target = (
    lambda name: print("Launching route-ffi" if name == "route_ffi" else "Launching depth-service")
)
bootstrap_mod.bootstrap = lambda one_click=False: None
sys.modules["solhunter_zero.bootstrap"] = bootstrap_mod

quick_setup_mod = types.ModuleType("scripts.quick_setup")
quick_setup_mod.main = lambda argv: _fake_select_keypair()
sys.modules["scripts.quick_setup"] = quick_setup_mod


def test_setup_one_click_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    script = Path("scripts/setup_one_click.py")
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run"])

    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)

    runpy.run_path(str(script), run_name="__main__")

    out = capsys.readouterr().out
    assert "Selected keypair: default" in out


def test_regenerates_proto_when_stale(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    script = Path("scripts/setup_one_click.py")
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run"])

    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)

    event_pb2 = Path("solhunter_zero/event_pb2.py")
    event_proto = Path("proto/event.proto")
    orig_times = (event_pb2.stat().st_atime, event_pb2.stat().st_mtime)
    os.utime(event_pb2, (event_proto.stat().st_atime, event_proto.stat().st_mtime - 10))

    called = {}

    def fake_check_call(cmd, *a, **k):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    runpy.run_path(str(script), run_name="__main__")

    assert "gen_proto.py" in " ".join(called["cmd"])
    os.utime(event_pb2, orig_times)


def test_sets_local_event_bus_url(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("EVENT_BUS_URL", raising=False)
    cfg_file = tmp_path / "config.toml"

    def fake_quick_setup(argv):
        cfg_file.write_text(
            'solana_rpc_url = "https://api.mainnet-beta.solana.com"\n'
            'dex_base_url = "https://quote-api.jup.ag"\n'
            'agents = ["simulation"]\n'
            "\n[agent_weights]\n"
            'simulation = 1.0\n'
        )

    qs = types.ModuleType("scripts.quick_setup")
    qs.main = fake_quick_setup
    qs.CONFIG_PATH = cfg_file
    monkeypatch.setitem(sys.modules, "scripts.quick_setup", qs)

    tomli_w_stub = types.ModuleType("tomli_w")

    def dumps(cfg):
        simple = {}
        tables = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                tables[k] = v
            else:
                simple[k] = v
        lines = []
        for k, v in simple.items():
            if isinstance(v, list):
                items = ", ".join(
                    f'"{i}"' if isinstance(i, str) else str(i) for i in v
                )
                lines.append(f"{k} = [{items}]")
            elif isinstance(v, str):
                lines.append(f"{k} = \"{v}\"")
            else:
                lines.append(f"{k} = {v}")
        for k, v in tables.items():
            lines.append(f"[{k}]")
            for sk, sv in v.items():
                if isinstance(sv, str):
                    lines.append(f"{sk} = \"{sv}\"")
                else:
                    lines.append(f"{sk} = {sv}")
        return "\n".join(lines)

    tomli_w_stub.dumps = dumps
    monkeypatch.setitem(sys.modules, "tomli_w", tomli_w_stub)

    monkeypatch.setattr(sys, "argv", ["scripts/setup_one_click.py"])
    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: None)

    runpy.run_path("scripts/setup_one_click.py", run_name="__main__")

    import tomllib

    cfg = tomllib.loads(cfg_file.read_text())
    assert cfg["event_bus_url"] == "ws://127.0.0.1:8787"
    assert os.environ["EVENT_BUS_URL"] == "ws://127.0.0.1:8787"
