import json
import os
import pytest
from solhunter_zero import wallet
from solders.keypair import Keypair


def setup_wallet(tmp_path, monkeypatch):
    monkeypatch.setattr(wallet, "KEYPAIR_DIR", str(tmp_path))
    monkeypatch.setattr(wallet, "ACTIVE_KEYPAIR_FILE", str(tmp_path / "active"))
    os.makedirs(wallet.KEYPAIR_DIR, exist_ok=True)


def test_load_keypair(tmp_path):
    kp = Keypair()
    path = tmp_path / "kp.json"
    path.write_text(json.dumps(list(kp.to_bytes())))
    loaded = wallet.load_keypair(str(path))
    assert loaded.to_bytes() == kp.to_bytes()


def test_save_and_list_keypairs(tmp_path, monkeypatch):
    setup_wallet(tmp_path, monkeypatch)
    data = [1, 2, 3]
    wallet.save_keypair("a", data)
    wallet.save_keypair("b", data)
    # create unrelated file
    (tmp_path / "other.txt").write_text("x")
    assert set(wallet.list_keypairs()) == {"a", "b"}
    assert json.loads((tmp_path / "a.json").read_text()) == data


def test_select_keypair_and_load_selected(tmp_path, monkeypatch):
    setup_wallet(tmp_path, monkeypatch)
    kp = Keypair()
    wallet.save_keypair("my", list(kp.to_bytes()))
    wallet.select_keypair("my")
    assert (tmp_path / "active").read_text() == "my"
    loaded = wallet.load_selected_keypair()
    assert loaded.to_bytes() == kp.to_bytes()


def test_load_selected_keypair_none(tmp_path, monkeypatch):
    setup_wallet(tmp_path, monkeypatch)
    assert wallet.load_selected_keypair() is None


def test_select_keypair_missing(tmp_path, monkeypatch):
    setup_wallet(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        wallet.select_keypair("missing")
