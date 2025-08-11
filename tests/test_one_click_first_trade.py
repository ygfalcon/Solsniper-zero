import sys
import os
import platform
import runpy
import types
import asyncio
from pathlib import Path

import pytest

# Stub out startup and supportive modules to avoid external actions
fake_startup = types.ModuleType("scripts.startup")
sys.modules["scripts.startup"] = fake_startup

macos_setup_mod = types.ModuleType("solhunter_zero.macos_setup")
macos_setup_mod.ensure_tools = lambda non_interactive=True: None
sys.modules["solhunter_zero.macos_setup"] = macos_setup_mod

bootstrap_utils_mod = types.ModuleType("solhunter_zero.bootstrap_utils")
bst = bootstrap_utils_mod
bst.ensure_venv = lambda arg: None
bst.ensure_deps = lambda cfg: None
bst.ensure_endpoints = lambda cfg: None
bst.ensure_cargo = lambda: None
bst.DepsConfig = lambda install_optional=False: None
bst.METAL_EXTRA_INDEX = []
sys.modules["solhunter_zero.bootstrap_utils"] = bst

logging_utils_mod = types.ModuleType("solhunter_zero.logging_utils")
logging_utils_mod.log_startup = lambda msg: None
logging_utils_mod.setup_logging = lambda name: None
sys.modules["solhunter_zero.logging_utils"] = logging_utils_mod

env_config_mod = types.ModuleType("solhunter_zero.env_config")
env_config_mod.configure_environment = lambda root: {}
env_config_mod.configure_startup_env = lambda root: {}
sys.modules["solhunter_zero.env_config"] = env_config_mod

device_mod = types.ModuleType("solhunter_zero.device")
device_mod.initialize_gpu = lambda: None
device_mod.METAL_EXTRA_INDEX = []
device_mod.get_default_device = lambda: "cpu"
device_mod.get_gpu_backend = lambda: "cpu"
device_mod.detect_gpu = lambda: None
sys.modules["solhunter_zero.device"] = device_mod

system_mod = types.ModuleType("solhunter_zero.system")
system_mod.set_rayon_threads = lambda: None
system_mod.detect_cpu_count = lambda: 1
sys.modules["solhunter_zero.system"] = system_mod

# Minimal stubs for pydantic and main module to avoid heavy imports
pydantic_mod = types.SimpleNamespace(
    BaseModel=object,
    AnyUrl=str,
    ValidationError=Exception,
    root_validator=lambda *a, **k: (lambda f: f),
    validator=lambda *a, **k: (lambda f: f),
    field_validator=lambda *a, **k: (lambda f: f),
    model_validator=lambda *a, **k: (lambda f: f),
)
sys.modules.setdefault("pydantic", pydantic_mod)

main_mod = types.ModuleType("solhunter_zero.main")


class _Trade:
    def __init__(self, token, side, amount, price):
        self.token = token
        self.side = side
        self.amount = amount
        self.price = price


class _Memory:
    def __init__(self, _):
        self._trades = []

    async def log_trade(self, token, side, amount, price):
        self._trades.append(_Trade(token, side, amount, price))

    async def list_trades(self):
        return self._trades


async def _run_auto(**kwargs):
    return 0

main_mod.Memory = _Memory
main_mod.run_auto = _run_auto
sys.modules["solhunter_zero.main"] = main_mod


def load_launcher():
    globals_dict = {
        "__name__": "scripts.launcher",
        "__file__": str(Path("scripts/launcher.py")),
    }
    runpy.run_path("scripts/launcher.py", globals_dict)
    import solhunter_zero.python_env as pyenv
    pyenv.find_python = lambda repair=False: sys.executable
    return sys.modules["solhunter_zero.launcher"]


def test_one_click_first_trade(monkeypatch, tmp_path):
    from solhunter_zero import main as main_module

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    dummy_device = types.ModuleType("device")
    dummy_device.initialize_gpu = lambda: None
    sys.modules["solhunter_zero.device"] = dummy_device
    if "solhunter_zero" in sys.modules:
        setattr(sys.modules["solhunter_zero"], "device", dummy_device)

    mem_path = tmp_path / "mem.db"
    pf_path = tmp_path / "pf.json"
    recorded: list = []

    def fake_run_auto(*, memory_path, portfolio_path, **kwargs):
        mem = main_module.Memory(memory_path)

        async def _do():
            await mem.log_trade(token="TOK", side="buy", amount=1, price=0)
            return await mem.list_trades()

        recorded.extend(asyncio.run(_do()))
        return 0

    monkeypatch.setattr(main_module, "run_auto", fake_run_auto)

    def _fake_run(argv):
        return main_module.run_auto(
            memory_path=f"sqlite:///{mem_path}",
            portfolio_path=str(pf_path),
        )

    fake_startup.run = _fake_run

    launcher = load_launcher()
    monkeypatch.setattr(launcher, "device", dummy_device, raising=False)

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

    assert len(recorded) == 1
    assert recorded[0].token == "TOK"
