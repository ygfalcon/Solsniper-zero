import os
import platform
import runpy
import sys
import types
import subprocess
import shutil
import site
from pathlib import Path

import pytest

# Stub modules to avoid heavy side effects during import
macos_setup_mod = types.ModuleType("solhunter_zero.macos_setup")
macos_setup_mod.ensure_tools = lambda non_interactive=True: None
macos_setup_mod._resolve_metal_versions = lambda: ("torch", "vision")
macos_setup_mod._write_versions_to_config = lambda *a, **k: None
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

# Stub wallet module to avoid heavy crypto imports
wallet_mod = types.ModuleType("solhunter_zero.wallet")
wallet_mod.setup_default_keypair = lambda: _fake_select_keypair()
sys.modules["solhunter_zero.wallet"] = wallet_mod


def test_setup_one_click_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    script = Path("scripts/setup_one_click.py")
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run"])

    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    Path(".env").write_text("")

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

    Path(".env").write_text("")

    runpy.run_path(str(script), run_name="__main__")

    assert "gen_proto.py" in " ".join(called["cmd"])
    os.utime(event_pb2, orig_times)


def test_single_trading_loop(monkeypatch):
    """setup_one_click should only start one trading loop."""

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    monkeypatch.delenv("AUTO_START", raising=False)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    calls: list[list[str]] = []

    class DummyPM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def start(self, cmd, *, stream_stderr=False):
            calls.append(cmd)
            return types.SimpleNamespace(poll=lambda: 0)

        def monitor_processes(self):
            pass

    def launch_services(pm):
        pm.start([sys.executable, "-m", "solhunter_zero.main"])

    def launch_ui(pm):
        if os.getenv("AUTO_START") == "1":
            pm.start([sys.executable, "-m", "solhunter_zero.main"])

    def main() -> None:
        with DummyPM() as pm:
            launch_services(pm)
            launch_ui(pm)
            pm.monitor_processes()

    start_all_stub = types.ModuleType("scripts.start_all")
    start_all_stub.ProcessManager = DummyPM
    start_all_stub.launch_services = launch_services
    start_all_stub.launch_ui = launch_ui
    start_all_stub.main = main
    sys.modules["scripts.start_all"] = start_all_stub

    monkeypatch.setattr(os, "execvp", lambda *a, **k: start_all_stub.main())

    script = Path("scripts/setup_one_click.py")
    Path(".env").write_text("")
    runpy.run_path(str(script), run_name="__main__")

    main_calls = [
        cmd for cmd in calls if cmd == [sys.executable, "-m", "solhunter_zero.main"]
    ]
    assert len(main_calls) == 1


def test_purges_corrupt_dist_info(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    script = Path("scripts/setup_one_click.py")
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run"])
    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    pkg_dir = site_dir / "solhunter_zero"
    pkg_dir.mkdir()
    dist_dir = site_dir / "solhunter_zero-0.1-invalid.dist-info"
    dist_dir.mkdir()

    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_dir)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(site_dir))

    def fake_run(cmd, *a, **k):
        if cmd[:3] == [sys.executable, "-m", "pip"] and cmd[3] == "show":
            out = f"Name: solhunter-zero\nVersion: invalid\nLocation: {site_dir}\n"
            return types.SimpleNamespace(returncode=0, stdout=out)
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    Path(".env").write_text("")
    runpy.run_path(str(script), run_name="__main__")

    assert not pkg_dir.exists()
    assert not dist_dir.exists()


def test_setup_one_click_creates_files(tmp_path, monkeypatch):
    """setup_one_click.main should create config and env files in a fresh repo."""
    import asyncio
    repo_root = tmp_path
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    # Ensure the script believes the repo root is tmp_path
    import solhunter_zero.paths as paths
    monkeypatch.setattr(paths, "ROOT", repo_root)

    # Prepare quick_setup stub that writes a minimal config
    cfg_path = repo_root / "config.toml"

    def fake_quick_setup(argv):
        cfg_path.write_text(
            "solana_rpc_url = \"https://api.mainnet-beta.solana.com\"\n"
            "dex_base_url = \"https://quote-api.jup.ag\"\n"
            "agents = [\"simulation\"]\n"
            "[agent_weights]\n"
            "simulation = 1.0\n"
        )

    quick_setup_mod.CONFIG_PATH = cfg_path
    monkeypatch.setattr(quick_setup_mod, "main", fake_quick_setup)

    # Wallet stub to create default keypair files
    def fake_wallet_setup_default_keypair():
        kp_dir = repo_root / "keypairs"
        kp_dir.mkdir()
        (kp_dir / "default.json").write_text("[]")
        return types.SimpleNamespace(name="default", mnemonic_path=None)

    monkeypatch.setattr(wallet_mod, "setup_default_keypair", fake_wallet_setup_default_keypair)

    # Avoid external subprocess calls and simulate proto generation
    def fake_check_call(cmd, *a, **k):
        if "gen_proto.py" in cmd[-1]:
            pkg_dir = repo_root / "solhunter_zero"
            pkg_dir.mkdir()
            (pkg_dir / "event_pb2.py").write_text("# generated")
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(os, "execvp", lambda *a, **k: None)

    # Create required proto directory
    (repo_root / "proto").mkdir()
    (repo_root / "proto" / "event.proto").write_text("syntax='proto3';")

    from scripts import setup_one_click

    setup_one_click.main(["--dry-run"])

    assert (repo_root / ".env").exists()
    assert cfg_path.exists()
    assert (repo_root / "keypairs" / "default.json").exists()
    assert (repo_root / "solhunter_zero" / "event_pb2.py").exists()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        assert not [t for t in asyncio.all_tasks(loop) if not t.done()]
