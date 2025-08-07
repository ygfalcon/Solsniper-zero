import os
import platform
import types
import sys

from scripts import startup


def test_startup_mac_m1(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})

    def fake_gpu_env():
        os.environ["SOLHUNTER_GPU_AVAILABLE"] = "0"
        os.environ["SOLHUNTER_GPU_DEVICE"] = "none"
        return {"SOLHUNTER_GPU_AVAILABLE": "0", "SOLHUNTER_GPU_DEVICE": "none"}

    dummy_device = types.SimpleNamespace(
        detect_gpu=lambda: True,
        get_default_device=lambda: "cpu",
        ensure_gpu_env=fake_gpu_env,
    )
    monkeypatch.setattr("solhunter_zero.device", dummy_device)
    monkeypatch.setattr(startup, "device", dummy_device)
    monkeypatch.setattr("solhunter_zero.bootstrap.device", dummy_device)

    dummy_torch = types.SimpleNamespace(set_default_device=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(startup, "torch", dummy_torch, raising=False)

    dummy_config = types.SimpleNamespace(
        load_config=lambda path: {},
        validate_config=lambda cfg: {},
        find_config_file=lambda: "config.toml",
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", dummy_config)

    monkeypatch.setattr(startup.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(startup, "ensure_depth_service", lambda: None)

    from solhunter_zero import bootstrap, wallet
    monkeypatch.setattr(bootstrap, "bootstrap", lambda one_click=False: None)
    monkeypatch.setattr(bootstrap, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_keypair", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_config", lambda: None)
    monkeypatch.setattr(wallet, "get_active_keypair_name", lambda: "default")
    monkeypatch.setattr(wallet, "list_keypairs", lambda: ["default"])

    monkeypatch.setattr("scripts.preflight.main", lambda: None)
    monkeypatch.setattr(startup.subprocess, "run", lambda cmd: types.SimpleNamespace(returncode=0))

    code = startup.run(["--one-click", "--self-test"])
    assert code == 0

