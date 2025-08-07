import os
import platform
import types
import sys
from pathlib import Path

from scripts import startup


def test_startup_mac_m1(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.ensure_tools", lambda: {"success": True}
    )

    def fake_gpu_env():
        os.environ["SOLHUNTER_GPU_AVAILABLE"] = "0"
        os.environ["SOLHUNTER_GPU_DEVICE"] = "cpu"
        os.environ["TORCH_DEVICE"] = "cpu"
        return {
            "SOLHUNTER_GPU_AVAILABLE": "0",
            "SOLHUNTER_GPU_DEVICE": "cpu",
            "TORCH_DEVICE": "cpu",
        }

    dummy_device = types.SimpleNamespace(
        detect_gpu=lambda: True,
        get_default_device=lambda: "cpu",
        ensure_gpu_env=fake_gpu_env,
        initialize_gpu=lambda: fake_gpu_env(),
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
    dummy_keypair_info = types.SimpleNamespace(name="default", mnemonic_path=None)
    monkeypatch.setattr(
        "solhunter_zero.config_utils.ensure_default_config",
        lambda: Path("config.toml"),
    )
    monkeypatch.setattr(
        "solhunter_zero.wallet_bootstrap.ensure_active_keypair",
        lambda one_click=True: (dummy_keypair_info, Path("keypairs/default.json")),
    )

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
    import solhunter_zero.wallet_bootstrap as wallet_bootstrap
    monkeypatch.setattr(wallet_bootstrap, "ensure_active_keypair", lambda one_click=False: (dummy_keypair_info, Path("dummy")))
    monkeypatch.setattr(bootstrap, "ensure_config", lambda: None)
    monkeypatch.setattr(wallet, "get_active_keypair_name", lambda: "default")
    monkeypatch.setattr(wallet, "list_keypairs", lambda: ["default"])

    monkeypatch.setattr(startup.subprocess, "run", lambda cmd: types.SimpleNamespace(returncode=0))
    logs: list[str] = []
    monkeypatch.setattr(startup, "log_startup", lambda msg: logs.append(msg))

    args = [
        "--skip-deps",
        "--skip-setup",
        "--skip-rpc-check",
        "--skip-endpoint-check",
        "--skip-preflight",
        "--no-diagnostics",
    ]
    code = startup.run(args)
    out = capsys.readouterr().out
    msg = "SolHunter Zero launch complete – system ready."
    assert msg in out
    assert msg in logs
    assert code == 0

