import asyncio
import json
import pytest

from solhunter_zero import depth_client


class FakeReader:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self):
        return self._data


class FakeWriter:
    def __init__(self):
        self.data = b""
        self.closed = False
        self.waited = False

    def write(self, data: bytes):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        self.waited = True


def test_snapshot(tmp_path, monkeypatch):
    data = {
        "tok": {
            "tx_rate": 2.5,
            "dex1": {"bids": 10, "asks": 5},
            "dex2": {"bids": 1},
        }
    }
    path = tmp_path / "depth.mmap"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(depth_client, "MMAP_PATH", str(path))

    venues, rate = depth_client.snapshot("tok")

    assert rate == pytest.approx(2.5)
    assert venues == {
        "dex1": {"bids": 10.0, "asks": 5.0},
        "dex2": {"bids": 1.0, "asks": 0.0},
    }


def test_submit_signed_tx(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        reader = FakeReader(json.dumps({"signature": "sig"}).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.submit_signed_tx("TX", socket_path="sock")

    sig = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {"cmd": "signed_tx", "tx": "TX"}
    assert sig == "sig"


def test_prepare_signed_tx(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        reader = FakeReader(json.dumps({"tx": "AAA"}).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.prepare_signed_tx(
            "MSG", priority_fee=7, socket_path="sock"
        )

    tx = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {"cmd": "prepare", "msg": "MSG", "priority_fee": 7}
    assert tx == "AAA"


def test_submit_tx_batch(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        reader = FakeReader(json.dumps(["a", "b"]).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.submit_tx_batch(
            ["t1", "t2"], socket_path="sock"
        )

    sigs = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {"cmd": "batch", "txs": ["t1", "t2"]}
    assert sigs == ["a", "b"]


def test_submit_raw_tx(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        reader = FakeReader(json.dumps({"signature": "sig"}).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.submit_raw_tx(
            "TX",
            socket_path="sock",
            priority_rpc=["u1", "u2"],
        )

    sig = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {
        "cmd": "raw_tx",
        "tx": "TX",
        "priority_rpc": ["u1", "u2"],
    }
    assert sig == "sig"


def test_auto_exec(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        reader = FakeReader(json.dumps({"ok": True}).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.auto_exec(
            "TOK",
            1.2,
            ["AA"],
            socket_path="sock",
        )

    res = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {
        "cmd": "auto_exec",
        "token": "TOK",
        "threshold": 1.2,
        "txs": ["AA"],
    }
    assert res is True
