import os
from pathlib import Path
import sys
import types

import pytest

# Provide minimal stubs for optional dependencies so the module can import
cryptography_fernet = types.SimpleNamespace(Fernet=object, InvalidToken=Exception)
sys.modules.setdefault("cryptography", types.SimpleNamespace(fernet=cryptography_fernet))
sys.modules.setdefault("cryptography.fernet", cryptography_fernet)

from solhunter_zero.env_config import configure_environment
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
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_check_required_env_missing(restore_env):
    os.environ.pop("SOLANA_RPC_URL", None)
    ok, msg = check_required_env()
    assert ok is False
    assert "SOLANA_RPC_URL" in msg


def test_check_required_env_placeholder(restore_env):
    os.environ["SOLANA_RPC_URL"] = "https://api.mainnet-beta.solana.com"
    os.environ["BIRDEYE_API_KEY"] = "YOUR_BIRDEYE_KEY"
    ok, msg = check_required_env()
    assert ok is True
    assert msg == "Required environment variables set"


def test_configure_env_strips_placeholder(tmp_path: Path, restore_env):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOLANA_RPC_URL=https://api.mainnet-beta.solana.com\n"
        "BIRDEYE_API_KEY=be_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )
    configure_environment(tmp_path)
    ok, msg = check_required_env()
    assert ok is True
    assert os.environ.get("BIRDEYE_API_KEY") == ""
    assert "be_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" not in env_file.read_text()


def test_configure_env_uses_config_value(tmp_path: Path, restore_env):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOLANA_RPC_URL=https://api.mainnet-beta.solana.com\n"
        "BIRDEYE_API_KEY=be_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )
    (tmp_path / "config.toml").write_text('birdeye_api_key="real_key"\n')
    configure_environment(tmp_path)
    ok, msg = check_required_env()
    assert ok is True
    assert os.environ.get("BIRDEYE_API_KEY") == "real_key"
    assert "real_key" in env_file.read_text()


def test_configure_env_creates_sanitized_env(tmp_path: Path, restore_env):
    example = tmp_path / ".env.example"
    example.write_text(
        "SOLANA_RPC_URL=https://api.mainnet-beta.solana.com\n"
        "BIRDEYE_API_KEY=be_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )
    configure_environment(tmp_path)
    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert "be_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" not in env_file.read_text()
    assert os.environ.get("BIRDEYE_API_KEY") == ""
