from solhunter_zero.preflight_utils import check_required_env


def test_check_required_env_without_birdeye(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://rpc")
    monkeypatch.delenv("BIRDEYE_API_KEY", raising=False)
    ok, msg = check_required_env()
    assert ok
    assert "Required environment variables" in msg


def test_check_required_env_with_birdeye(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://rpc")
    monkeypatch.delenv("BIRDEYE_API_KEY", raising=False)
    ok, msg = check_required_env(["SOLANA_RPC_URL", "BIRDEYE_API_KEY"])
    assert not ok
    assert "BIRDEYE_API_KEY" in msg
    monkeypatch.setenv("BIRDEYE_API_KEY", "abc")
    ok, msg = check_required_env(["SOLANA_RPC_URL", "BIRDEYE_API_KEY"])
    assert ok
