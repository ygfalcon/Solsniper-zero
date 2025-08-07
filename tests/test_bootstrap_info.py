import types
from pathlib import Path

from solhunter_zero import bootstrap


def test_bootstrap_returns_setup_info(monkeypatch, tmp_path):
    monkeypatch.delenv("SOLHUNTER_SKIP_SETUP", raising=False)
    monkeypatch.setattr(bootstrap, "ensure_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(bootstrap, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(bootstrap.device, "ensure_gpu_env", lambda: {})

    kp_info = types.SimpleNamespace(name="alice", mnemonic_path=tmp_path / "mnemonic.txt")

    def fake_keypair():
        return kp_info, tmp_path / "alice.json"

    monkeypatch.setattr(bootstrap, "ensure_keypair", fake_keypair)
    monkeypatch.setattr(bootstrap, "ensure_config", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(bootstrap.wallet, "ensure_default_keypair", lambda: kp_info)

    info = bootstrap.bootstrap(one_click=True)

    assert info["config_path"] == tmp_path / "config.toml"
    assert info["keypair_path"] == tmp_path / "alice.json"
    assert info["active_keypair"] == "alice"
    assert info["mnemonic_path"] == tmp_path / "mnemonic.txt"
