import json
from solders.keypair import Keypair
from solhunter_zero.keypair import load_keypair


def test_load_keypair(tmp_path):
    kp = Keypair()
    path = tmp_path / "kp.json"
    path.write_text(json.dumps(list(bytes(kp))))
    loaded = load_keypair(path)
    assert loaded.pubkey() == kp.pubkey()
