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


def test_check_required_env_success(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://rpc.example"
    os.environ["BIRDEYE_API_KEY"] = "real_key"
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_check_required_env_missing(restore_env):
    os.environ.pop("SOLANA_RPC_URL", None)
    os.environ.pop("BIRDEYE_API_KEY", None)
    ok, msg = check_required_env()
    assert ok is False
    assert "SOLANA_RPC_URL" in msg and "BIRDEYE_API_KEY" in msg


def test_check_required_env_placeholder(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://rpc.example"
    os.environ["BIRDEYE_API_KEY"] = "YOUR_BIRDEYE_KEY"
    ok, msg = check_required_env()
    assert ok is False
    assert msg.startswith("Missing environment variables: BIRDEYE_API_KEY")
