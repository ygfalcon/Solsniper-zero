import runpy
import sys
from pathlib import Path
import types


def test_start_all_imports(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    bu = types.ModuleType("solhunter_zero.bootstrap_utils")
    sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))
    solders = types.ModuleType("solders")
    keypair = types.ModuleType("keypair")
    keypair.Keypair = type("Keypair", (), {})
    solders.keypair = keypair  # type: ignore[attr-defined]
    sys.modules["solders"] = solders
    sys.modules["solders.keypair"] = keypair
    bip_utils = types.ModuleType("bip_utils")
    bip_utils.Bip39SeedGenerator = bip_utils.Bip44 = bip_utils.Bip44Coins = bip_utils.Bip44Changes = object
    sys.modules["bip_utils"] = bip_utils
    autopilot = types.ModuleType("solhunter_zero.autopilot")
    autopilot._maybe_start_event_bus = lambda cfg: None
    autopilot.shutdown_event_bus = lambda: None
    sys.modules["solhunter_zero.autopilot"] = autopilot
    ui = types.ModuleType("solhunter_zero.ui")
    ui.rl_ws_loop = ui.event_ws_loop = ui.log_ws_loop = None
    ui.create_app = lambda: None
    ui.start_websockets = lambda: {}
    sys.modules["solhunter_zero.ui"] = ui
    bootstrap_module = types.ModuleType("solhunter_zero.bootstrap")
    bootstrap_module.bootstrap = lambda one_click=True: None
    bootstrap_module.ensure_keypair = lambda: None
    sys.modules["solhunter_zero.bootstrap"] = bootstrap_module

    def ensure_venv(argv=None):
        return None

    def ensure_cargo(*a, **k):
        return None

    def prepend_repo_root() -> None:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    bu.ensure_venv = ensure_venv
    bu.ensure_cargo = ensure_cargo
    bu.prepend_repo_root = prepend_repo_root
    sys.modules["solhunter_zero.bootstrap_utils"] = bu

    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setenv("SOLHUNTER_TESTING", "1")

    sys.modules.pop("solhunter_zero.device", None)
    original_path = list(sys.path)
    sys.path[:] = [str(scripts_dir)] + [p for p in original_path if p not in (str(root), str(scripts_dir))]
    monkeypatch.chdir(scripts_dir)

    runpy.run_path("start_all.py", run_name="not_main")

    assert "solhunter_zero.device" in sys.modules
