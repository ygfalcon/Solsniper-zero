import time
import urllib.request
from urllib.error import URLError

import pytest

from solhunter_zero import preflight_utils


class DummyResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"ok"


def test_check_disk_space_success(monkeypatch):
    def fake_disk_usage(path):
        return (100, 50, 20)

    monkeypatch.setattr(preflight_utils.shutil, "disk_usage", fake_disk_usage)
    assert preflight_utils.check_disk_space(10) is None


def test_check_disk_space_failure(monkeypatch, capsys):
    def fake_disk_usage(path):
        return (100, 50, 5)

    monkeypatch.setattr(preflight_utils.shutil, "disk_usage", fake_disk_usage)
    with pytest.raises(SystemExit):
        preflight_utils.check_disk_space(10)
    out = capsys.readouterr().out
    assert "Insufficient disk space" in out


def test_check_internet_success(monkeypatch):
    def fake_urlopen(url, timeout=5):
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    assert preflight_utils.check_internet("https://example.com") is None


def test_check_internet_failure(monkeypatch, capsys):
    def fake_urlopen(url, timeout=5):
        raise URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with pytest.raises(SystemExit):
        preflight_utils.check_internet("https://example.com")
    out = capsys.readouterr().out
    assert "Failed to reach https://example.com after 3 attempts" in out


def test_check_required_env_placeholder_birdeye(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "https://example.com")
    monkeypatch.setenv("BIRDEYE_API_KEY", "BD1234567890ABCDEFGHIJKL")
    ok, msg = preflight_utils.check_required_env()
    assert ok is False
    assert "BIRDEYE_API_KEY" in msg


def test_check_required_env_placeholder_jito(monkeypatch):
    monkeypatch.setenv("JITO_AUTH", "YOUR_JITO_AUTH_TOKEN")
    ok, msg = preflight_utils.check_required_env(["JITO_AUTH"])
    assert ok is False
    assert "JITO_AUTH" in msg
