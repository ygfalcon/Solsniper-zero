import base64
import asyncio
import pytest

from solders.keypair import Keypair
from solders.hash import Hash
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from solhunter_zero.exchange import place_order, place_order_async


async def _no_fee_async(*a, **k):
    return 0.0


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = "resp"

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("bad status")

    def json(self):
        return self._data


def _dummy_tx(kp: Keypair) -> str:
    msg = MessageV0.try_compile(
        kp.pubkey(), [Instruction(Pubkey.default(), b"", [])], [], Hash.new_unique()
    )
    tx = VersionedTransaction.populate(msg, [Signature.default()])
    return base64.b64encode(bytes(tx)).decode()


def test_place_order_sends(monkeypatch):
    kp = Keypair()
    sent = {}

    def fake_post(url, json, timeout=10):
        sent["url"] = url
        return FakeResponse({"swapTransaction": _dummy_tx(kp)})

    class FakeClient:
        def __init__(self, url):
            sent["rpc"] = url

        def send_raw_transaction(self, data, opts=None):
            sent["data_len"] = len(data)

            class Resp:
                value = "sig"

            return Resp()

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    monkeypatch.setattr("solhunter_zero.exchange.Client", FakeClient)
    result = place_order("tok", "buy", 1.0, 0.0, keypair=kp, testnet=True)
    assert result["signature"] == "sig"
    assert sent["data_len"] > 0
    assert "/v6/swap" in sent["url"]


def test_place_order_dry_run(monkeypatch):
    kp = Keypair()
    called = {}

    def fake_post(*a, **k):
        called["post"] = True
        return FakeResponse({})

    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    result = place_order("tok", "buy", 1.0, 0.0, keypair=kp, dry_run=True)
    assert result["dry_run"] is True
    assert "post" not in called


def test_place_order_async(monkeypatch):
    kp = Keypair()
    sent = {}

    class FakeResp:
        def __init__(self, url):
            sent["url"] = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {"swapTransaction": _dummy_tx(kp)}

        def raise_for_status(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json, timeout=10):
            return FakeResp(url)

    class FakeClient:
        def __init__(self, url):
            sent["rpc"] = url

        async def send_raw_transaction(self, data, opts=None):
            sent["len"] = len(data)

            class Resp:
                value = "sig"

            return Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    monkeypatch.setattr("solhunter_zero.exchange.AsyncClient", FakeClient)
    monkeypatch.setattr("solhunter_zero.exchange.get_current_fee_async", _no_fee_async)
    monkeypatch.setattr("solhunter_zero.exchange.USE_RUST_EXEC", False)
    result = asyncio.run(place_order_async("tok", "buy", 1.0, 0.0, keypair=kp, testnet=True))
    assert result["signature"] == "sig"
    assert sent["len"] > 0


def test_place_order_async_deducts_gas(monkeypatch):
    kp = Keypair()
    sent = {}

    class FakeResp:
        def __init__(self, url, data):
            sent["url"] = url
            sent["payload"] = data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {"swapTransaction": _dummy_tx(kp)}

        def raise_for_status(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def post(self, url, json, timeout=10):
            return FakeResp(url, json)

    class FakeClient:
        def __init__(self, url):
            pass

        async def send_raw_transaction(self, data, opts=None):
            class Resp:
                value = "sig"

            return Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession())
    monkeypatch.setattr("solhunter_zero.exchange.AsyncClient", FakeClient)
    async def fake_fee(*a, **k):
        return 1.0
    monkeypatch.setattr("solhunter_zero.exchange.get_current_fee_async", fake_fee)
    monkeypatch.setattr("solhunter_zero.exchange.USE_RUST_EXEC", False)

    asyncio.run(place_order_async("tok", "buy", 2.0, 0.0, keypair=kp))
    assert sent["payload"]["amount"] == pytest.approx(1.0)


def test_place_order_reuses_loop(monkeypatch):
    kp = Keypair()
    loops: list[asyncio.AbstractEventLoop] = []

    orig_new_event_loop = asyncio.new_event_loop

    def fake_new_event_loop():
        loop = orig_new_event_loop()
        loops.append(loop)
        return loop

    def fake_post(url, json, timeout=10):
        return FakeResponse({"swapTransaction": _dummy_tx(kp)})

    class FakeClient:
        def __init__(self, url):
            pass

        def send_raw_transaction(self, data, opts=None):
            class Resp:
                value = "sig"

            return Resp()

    monkeypatch.setattr(asyncio, "new_event_loop", fake_new_event_loop)
    monkeypatch.setattr("solhunter_zero.exchange.requests.post", fake_post)
    monkeypatch.setattr("solhunter_zero.exchange.Client", FakeClient)
    monkeypatch.setattr("solhunter_zero.exchange._order_loop", None, raising=False)

    place_order("tok", "buy", 1.0, 0.0, keypair=kp)
    place_order("tok", "buy", 1.0, 0.0, keypair=kp)

    assert len(loops) == 1
