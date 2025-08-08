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
    os.environ["SOLANA_RPC_URL"] = "https://api.mainnet-beta.solana.com"
    os.environ["BIRDEYE_API_KEY"] = "real_key"
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_check_required_env_rpc_only(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://api.mainnet-beta.solana.com"
    os.environ.pop("BIRDEYE_API_KEY", None)
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_check_required_env_birdeye_only(restore_env):
    os.environ.pop("SOLANA_RPC_URL", None)
    os.environ["BIRDEYE_API_KEY"] = "bd_real_key"
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_check_required_env_missing(restore_env):
    os.environ.pop("SOLANA_RPC_URL", None)
    os.environ.pop("BIRDEYE_API_KEY", None)
    ok, msg = check_required_env()
    assert ok is False
    assert "BIRDEYE_API_KEY or SOLANA_RPC_URL" in msg


@pytest.mark.parametrize(
    "placeholder",
    [
        "YOUR_BIRDEYE_KEY",
        "be_" + "X" * 32,
        "BD" + "1234567890" * 3,
    ],
)
def test_check_required_env_placeholder_patterns(restore_env, placeholder):
    os.environ["BIRDEYE_API_KEY"] = placeholder
    ok, msg = check_required_env(keys=["BIRDEYE_API_KEY"])
    assert ok is False
    assert msg.startswith("Missing environment variables: BIRDEYE_API_KEY")

