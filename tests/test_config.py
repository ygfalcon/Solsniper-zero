import os
from solhunter_zero.config import load_config, apply_env_overrides, set_env_from_config


def test_load_config_yaml(tmp_path):
    path = tmp_path / "my.yaml"
    path.write_text("birdeye_api_key: KEY\nsolana_rpc_url: http://local")
    cfg = load_config(str(path))
    assert cfg == {"birdeye_api_key": "KEY", "solana_rpc_url": "http://local"}


def test_load_config_toml(tmp_path):
    path = tmp_path / "my.toml"
    path.write_text('birdeye_api_key="KEY"\nsolana_rpc_url="http://local"')
    cfg = load_config(str(path))
    assert cfg == {"birdeye_api_key": "KEY", "solana_rpc_url": "http://local"}


def test_env_var_overrides_default_search(tmp_path, monkeypatch):
    default = tmp_path / "config.yaml"
    default.write_text("birdeye_api_key: DEFAULT")
    override = tmp_path / "ov.toml"
    override.write_text('birdeye_api_key="OVR"')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOLHUNTER_CONFIG", str(override))
    cfg = load_config()
    assert cfg == {"birdeye_api_key": "OVR"}


def test_apply_env_overrides(monkeypatch):
    cfg = {"birdeye_api_key": "a", "solana_rpc_url": "b"}
    monkeypatch.setenv("BIRDEYE_API_KEY", "NEW")
    result = apply_env_overrides(cfg)
    assert result["birdeye_api_key"] == "NEW"
    assert result["solana_rpc_url"] == "b"


def test_set_env_from_config(monkeypatch):
    cfg = {"birdeye_api_key": "A", "solana_rpc_url": "RPC"}
    monkeypatch.delenv("BIRDEYE_API_KEY", raising=False)
    monkeypatch.setenv("SOLANA_RPC_URL", "EXIST")
    set_env_from_config(cfg)
    assert os.getenv("BIRDEYE_API_KEY") == "A"
    assert os.getenv("SOLANA_RPC_URL") == "EXIST"
