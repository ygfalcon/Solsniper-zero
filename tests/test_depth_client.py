import asyncio
import json
import struct
import pytest

from solhunter_zero import depth_client


@pytest.fixture(autouse=True)
def _reset_pool():
    asyncio.run(depth_client.close_ipc_clients())
    yield
    asyncio.run(depth_client.close_ipc_clients())


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

    def is_closing(self):
        return self.closed

    async def wait_closed(self):
        self.waited = True


def build_index(path, entries):
    header = bytearray(b"IDX1")
    header.extend(struct.pack("<I", len(entries)))
    header_size = 8 + sum(2 + len(k) + 8 for k in entries)
    data = bytearray()
    for token, obj in entries.items():
        b = json.dumps(obj).encode()
        header.extend(struct.pack("<H", len(token)))
        header.extend(token.encode())
        header.extend(struct.pack("<II", header_size + len(data), len(b)))
        data.extend(b)
    path.write_bytes(bytes(header + data))


def test_snapshot(tmp_path, monkeypatch):
    data = {
        "tok": {
            "tx_rate": 2.5,
            "dex1": {"bids": 10, "asks": 5},
            "dex2": {"bids": 1},
        }
    }
    path = tmp_path / "depth.mmap"
    build_index(path, data)
    monkeypatch.setattr(depth_client, "MMAP_PATH", str(path))

    venues, rate = depth_client.snapshot("tok")

    assert rate == pytest.approx(2.5)
    assert venues == {
        "dex1": {"bids": 10.0, "asks": 5.0},
        "dex2": {"bids": 1.0, "asks": 0.0},
    }


def test_snapshot_cache(tmp_path, monkeypatch):
    data = {
        "tok": {"tx_rate": 1.0, "dex": {"bids": 1, "asks": 1}}
    }
    path = tmp_path / "depth.mmap"
    build_index(path, data)
    monkeypatch.setattr(depth_client, "MMAP_PATH", str(path))
    depth_client.SNAPSHOT_CACHE.clear()
    monkeypatch.setattr(depth_client, "DEPTH_CACHE_TTL", 0.5)

    import builtins

    calls = []

    original_open = builtins.open

    def fake_open(file, *args, **kwargs):
        if file == str(path):
            calls.append(1)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    times = [0.0, 0.1]

    def fake_monotonic():
        return times.pop(0)

    monkeypatch.setattr(depth_client.time, "monotonic", fake_monotonic)

    venues1, rate1 = depth_client.snapshot("tok")
    venues2, rate2 = depth_client.snapshot("tok")
    monkeypatch.setattr(builtins, "open", original_open)

    assert venues1 == venues2
    assert rate1 == rate2
    assert len(calls) == 1


def test_snapshot_json_fallback(tmp_path, monkeypatch):
    data = {
        "tok": {"tx_rate": 3.0, "dex": {"bids": 2, "asks": 4}}
    }
    path = tmp_path / "depth.mmap"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(depth_client, "MMAP_PATH", str(path))
    depth_client.SNAPSHOT_CACHE.clear()

    venues, rate = depth_client.snapshot("tok")

    assert rate == pytest.approx(3.0)
    assert venues == {"dex": {"bids": 2.0, "asks": 4.0}}


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
            priority_fee=7,
        )

    sig = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {
        "cmd": "raw_tx",
        "tx": "TX",
        "priority_rpc": ["u1", "u2"],
        "priority_fee": 7,
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


def test_best_route(monkeypatch):
    captured = {}

    async def fake_conn(path):
        captured["socket"] = path
        writer = FakeWriter()
        resp = {"path": ["a", "b"], "profit": 1.0, "slippage": 0.2}
        reader = FakeReader(json.dumps(resp).encode())
        captured["writer"] = writer
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        return await depth_client.best_route(
            "TOK", 2.0, socket_path="sock", max_hops=4
        )

    res = asyncio.run(run())
    payload = json.loads(captured["writer"].data.decode())

    assert captured["socket"] == "sock"
    assert payload == {
        "cmd": "route",
        "token": "TOK",
        "amount": 2.0,
        "max_hops": 4,
    }
    assert res == (["a", "b"], 1.0, 0.2)


def test_listen_depth_ws(monkeypatch):
    msgs = [{"tok": {"bids": 1, "asks": 2, "tx_rate": 1.0}}]

    class FakeMsg:
        def __init__(self, data):
            self.type = aiohttp.WSMsgType.TEXT
            self.data = json.dumps(data)

    class FakeWS:
        def __init__(self, messages):
            self.messages = list(messages)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.messages:
                return FakeMsg(self.messages.pop(0))
            raise StopAsyncIteration

    class FakeSession:
        def __init__(self, messages):
            self.messages = messages

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def ws_connect(self, url):
            self.url = url
            return FakeWS(self.messages)

    monkeypatch.setattr("aiohttp.ClientSession", lambda: FakeSession(msgs))
    events = []
    monkeypatch.setattr(depth_client, "publish", lambda t, d: events.append((t, d)))

    asyncio.run(depth_client.listen_depth_ws(max_updates=1))

    assert events[0][0] == "depth_service_status"
    assert events[0][1]["status"] == "connected"
    assert events[1] == ("depth_update", msgs[0])
    assert events[2][0] == "depth_service_status" and events[2][1]["status"] == "disconnected"


def test_connection_pool_reuse(monkeypatch):
    calls = []

    async def fake_conn(path):
        calls.append(path)
        writer = FakeWriter()
        reader = FakeReader(json.dumps({"ok": True}).encode())
        return reader, writer

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_conn)

    async def run():
        await depth_client.auto_exec("TOK", 1.0, ["A"], socket_path="sock")
        await depth_client.best_route("TOK", 1.0, socket_path="sock")

    asyncio.run(run())

    assert calls == ["sock"]
