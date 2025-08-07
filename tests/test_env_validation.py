import json
import os
import json
import os
import pytest
import tempfile

_cfg = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
_cfg.write("solana_rpc_url='http://localhost'\n")
_cfg.write("dex_base_url='http://localhost'\n")
_cfg.write("agents=['dummy']\n")
_cfg.write("agent_weights={dummy=1.0}\n")
_cfg.flush()
os.environ.setdefault("SOLHUNTER_CONFIG", _cfg.name)

os.environ.setdefault("SOLANA_RPC_URL", "http://localhost")
os.environ.setdefault("DEX_BASE_URL", "http://localhost")
os.environ.setdefault("AGENTS", "[]")

from solhunter_zero import main

pytest.importorskip("solders")


def _write_keypair(path):
    path.write_text(json.dumps(list(range(64))))


def test_missing_solana_rpc_url(tmp_path, monkeypatch):
    key_path = tmp_path / "kp.json"
    _write_keypair(key_path)
    monkeypatch.setenv("KEYPAIR_PATH", str(key_path))
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    with pytest.raises(RuntimeError, match="SOLANA_RPC_URL is not set"):
        main.validate_environment()


def test_malformed_solana_rpc_url(tmp_path, monkeypatch):
    key_path = tmp_path / "kp.json"
    _write_keypair(key_path)
    monkeypatch.setenv("KEYPAIR_PATH", str(key_path))
    monkeypatch.setenv("SOLANA_RPC_URL", "not a url")
    with pytest.raises(RuntimeError, match="SOLANA_RPC_URL 'not a url' is malformed"):
        main.validate_environment()


def test_missing_keypair_path(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "https://example.com")
    monkeypatch.delenv("KEYPAIR_PATH", raising=False)
    monkeypatch.delenv("SOLANA_KEYPAIR", raising=False)
    with pytest.raises(RuntimeError, match="KEYPAIR_PATH is not set"):
        main.validate_environment()


def test_invalid_keypair_file(tmp_path, monkeypatch):
    bad_path = tmp_path / "kp.json"
    bad_path.write_text("not json")
    monkeypatch.setenv("SOLANA_RPC_URL", "https://example.com")
    monkeypatch.setenv("KEYPAIR_PATH", str(bad_path))
    with pytest.raises(RuntimeError, match="Invalid keypair"):
        main.validate_environment()
