import json
from solhunter_zero.wallet import load_keypair
from solders.keypair import Keypair


def test_load_keypair(tmp_path):
    kp = Keypair()
    path = tmp_path / "kp.json"
    path.write_text(json.dumps(list(kp.to_bytes())))
    loaded = load_keypair(str(path))
    assert loaded.to_bytes() == kp.to_bytes()
