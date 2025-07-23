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


def test_load_config_agents(tmp_path):
    path = tmp_path / "agents.toml"
    path.write_text(
        'agents=["sim","exit"]\n[agent_weights]\nsim=0.5\nexit=1.0\n'
    )
    cfg = load_config(str(path))
    assert cfg["agents"] == ["sim", "exit"]
    assert cfg["agent_weights"] == {"sim": 0.5, "exit": 1.0}


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
    cfg = {
        "birdeye_api_key": "a",
        "solana_rpc_url": "b",
        "risk_tolerance": 0.1,
        "token_suffix": "bonk",
        "agents": ["a"],
        "agent_weights": {"a": 1.0},
    }
    monkeypatch.setenv("BIRDEYE_API_KEY", "NEW")
    monkeypatch.setenv("RISK_TOLERANCE", "0.2")
    monkeypatch.setenv("TOKEN_SUFFIX", "doge")
    monkeypatch.setenv("AGENTS", "x,y")
    monkeypatch.setenv("AGENT_WEIGHTS", "{'x': 1}")
    result = apply_env_overrides(cfg)
    assert result["birdeye_api_key"] == "NEW"
    assert result["solana_rpc_url"] == "b"
    assert result["risk_tolerance"] == "0.2"
    assert result["token_suffix"] == "doge"
    assert result["agents"] == "x,y"
    assert result["agent_weights"] == "{'x': 1}"


def test_set_env_from_config(monkeypatch):
    cfg = {
        "birdeye_api_key": "A",
        "solana_rpc_url": "RPC",
        "risk_tolerance": 0.3,
        "token_suffix": "xyz",
        "agents": ["sim"],
        "agent_weights": {"sim": 1.0},
    }
    monkeypatch.delenv("BIRDEYE_API_KEY", raising=False)
    monkeypatch.delenv("RISK_TOLERANCE", raising=False)
    monkeypatch.delenv("TOKEN_SUFFIX", raising=False)
    monkeypatch.delenv("AGENTS", raising=False)
    monkeypatch.delenv("AGENT_WEIGHTS", raising=False)
    monkeypatch.setenv("SOLANA_RPC_URL", "EXIST")
    set_env_from_config(cfg)
    assert os.getenv("BIRDEYE_API_KEY") == "A"
    assert os.getenv("SOLANA_RPC_URL") == "EXIST"
    assert os.getenv("RISK_TOLERANCE") == "0.3"
    assert os.getenv("TOKEN_SUFFIX") == "xyz"
    assert os.getenv("AGENTS") == "['sim']"
    assert os.getenv("AGENT_WEIGHTS") == "{'sim': 1.0}"


def test_set_env_from_config_booleans(monkeypatch):
    cfg = {
        "use_flash_loans": True,
        "use_depth_stream": True,
        "use_rust_exec": True,
        "use_service_exec": True,
        "use_mev_bundles": True,
    }
    monkeypatch.delenv("USE_FLASH_LOANS", raising=False)
    monkeypatch.delenv("USE_DEPTH_STREAM", raising=False)
    monkeypatch.delenv("USE_RUST_EXEC", raising=False)
    monkeypatch.delenv("USE_SERVICE_EXEC", raising=False)
    monkeypatch.delenv("USE_MEV_BUNDLES", raising=False)
    set_env_from_config(cfg)
    assert os.getenv("USE_FLASH_LOANS") == "True"
    assert os.getenv("USE_DEPTH_STREAM") == "True"
    assert os.getenv("USE_RUST_EXEC") == "True"
    assert os.getenv("USE_SERVICE_EXEC") == "True"
    assert os.getenv("USE_MEV_BUNDLES") == "True"
