import os
import pytest

from solhunter_zero.preflight_utils import check_required_env


@pytest.fixture
def restore_env():
    """Snapshot os.environ and restore after the test."""
    original = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def test_check_required_env_warns_without_birdeye(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://api.mainnet-beta.solana.com"
    ok, msg = check_required_env()
    assert ok is True
    assert "BIRDEYE_API_KEY" in msg


def test_check_required_env_missing_rpc(restore_env):
    os.environ.pop("SOLANA_RPC_URL", None)
    ok, msg = check_required_env()
    assert ok is False
    assert "SOLANA_RPC_URL" in msg


def test_check_required_env_require_birdeye(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://api.mainnet-beta.solana.com"
    ok, msg = check_required_env(require_birdeye=True)
    assert ok is False
    assert "BIRDEYE_API_KEY" in msg
