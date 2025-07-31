import os
import sys
import types
import importlib.util

if importlib.util.find_spec("aiohttp") is None:
    aiohttp_mod = types.ModuleType("aiohttp")
    aiohttp_mod.ClientSession = object
    aiohttp_mod.TCPConnector = object
    sys.modules["aiohttp"] = aiohttp_mod

# Stub other optional dependencies when unavailable
if importlib.util.find_spec("transformers") is None:
    trans_mod = types.ModuleType("transformers")
    trans_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    trans_mod.pipeline = lambda *a, **k: lambda *a2, **k2: None
    sys.modules["transformers"] = trans_mod

if importlib.util.find_spec("sentence_transformers") is None:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    sys.modules.setdefault("sentence_transformers", st_mod)


try:
    _gp_spec = importlib.util.find_spec("google.protobuf")
except ModuleNotFoundError:
    _gp_spec = None
if _gp_spec is None:
    protobuf = types.ModuleType("protobuf")
    descriptor = types.ModuleType("descriptor")
    descriptor_pool = types.ModuleType("descriptor_pool")
    symbol_database = types.ModuleType("symbol_database")
    symbol_database.Default = lambda: object()
    internal = types.ModuleType("internal")
    internal.builder = types.ModuleType("builder")
    runtime_version = types.ModuleType("runtime_version")
    runtime_version.ValidateProtobufRuntimeVersion = lambda *a, **k: None
    protobuf.descriptor = descriptor
    protobuf.descriptor_pool = descriptor_pool
    protobuf.symbol_database = symbol_database
    protobuf.internal = internal
    protobuf.runtime_version = runtime_version
    try:
        import google
    except ModuleNotFoundError:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules.setdefault("google", google)
    google.protobuf = protobuf
    sys.modules.setdefault("google.protobuf", protobuf)
    sys.modules.setdefault("google.protobuf.descriptor", descriptor)
    sys.modules.setdefault("google.protobuf.descriptor_pool", descriptor_pool)
    sys.modules.setdefault("google.protobuf.symbol_database", symbol_database)
    sys.modules.setdefault("google.protobuf.internal", internal)
    sys.modules.setdefault("google.protobuf.internal.builder", internal.builder)
    sys.modules.setdefault("google.protobuf.runtime_version", runtime_version)

event_pb2 = types.ModuleType("event_pb2")
for name in [
    "ActionExecuted",
    "WeightsUpdated",
    "RLWeights",
    "RLCheckpoint",
    "PortfolioUpdated",
    "DepthUpdate",
    "DepthServiceStatus",
    "Heartbeat",
    "TradeLogged",
    "RLMetrics",
    "PriceUpdate",
    "SystemMetrics",
    "SystemMetricsCombined",
    "RiskMetrics",
    "RiskUpdated",
    "RemoteSystemMetrics",
    "TokenDiscovered",
    "MemorySyncRequest",
    "MemorySyncResponse",
    "PendingSwap",
    "ConfigUpdated",
    "Event",
]:
    setattr(event_pb2, name, type(name, (), {}))
sys.modules.setdefault("solhunter_zero.event_pb2", event_pb2)
sys.modules.setdefault("event_pb2", event_pb2)

if importlib.util.find_spec("solders") is None:
    solders_mod = types.ModuleType("solders")
    solders_mod.__spec__ = importlib.machinery.ModuleSpec("solders", None)
    sys.modules["solders"] = solders_mod
    kp_mod = types.ModuleType("solders.keypair")

    class Keypair:
        def __init__(self, data: bytes | None = None):
            self._data = data or b"\x00" * 64

        @classmethod
        def from_bytes(cls, b: bytes):
            return cls(bytes(b))

        @classmethod
        def from_seed(cls, seed: bytes):
            return cls(seed.ljust(64, b"\0"))

        def to_bytes(self) -> bytes:
            return self._data

        def to_bytes_array(self) -> list[int]:
            return list(self._data)

        def sign_message(self, _msg: bytes) -> bytes:
            return b"sig"

        def pubkey(self) -> str:
            return "0" * 32

    kp_mod.Keypair = Keypair
    kp_mod.__spec__ = importlib.machinery.ModuleSpec("solders.keypair", None)
    sys.modules["solders.keypair"] = kp_mod
    for name in ["pubkey", "hash", "message", "transaction", "instruction", "signature"]:
        mod = types.ModuleType(f"solders.{name}")
        mod.__spec__ = importlib.machinery.ModuleSpec(f"solders.{name}", None)
        sys.modules.setdefault(f"solders.{name}", mod)
    sys.modules["solders.pubkey"].Pubkey = object
    sys.modules["solders.hash"].Hash = object
    sys.modules["solders.message"].MessageV0 = object
    sys.modules["solders.transaction"].VersionedTransaction = type("VersionedTransaction", (), {"populate": staticmethod(lambda *a, **k: object())})
    sys.modules["solders.instruction"].Instruction = object
    sys.modules["solders.instruction"].AccountMeta = object
    sys.modules["solders.signature"].Signature = type("Signature", (), {"default": staticmethod(lambda: object())})

if importlib.util.find_spec("solana") is None:
    sol_mod = types.ModuleType("solana")
    sol_mod.__spec__ = importlib.machinery.ModuleSpec("solana", None)
    sys.modules["solana"] = sol_mod
    pub_mod = types.ModuleType("solana.publickey")
    pub_mod.__spec__ = importlib.machinery.ModuleSpec("solana.publickey", None)

    class PublicKey:
        def __init__(self, value: str | None = None):
            self.value = value

        @staticmethod
        def default():
            return PublicKey("0" * 32)

        @staticmethod
        def from_string(val: str):
            return PublicKey(val)

    pub_mod.PublicKey = PublicKey
    sys.modules["solana.publickey"] = pub_mod
    rpc_mod = types.ModuleType("solana.rpc")
    rpc_mod.__spec__ = importlib.machinery.ModuleSpec("solana.rpc", None)
    sys.modules.setdefault("solana.rpc", rpc_mod)
    sys.modules.setdefault("solana.rpc.api", types.SimpleNamespace(Client=object, __spec__=importlib.machinery.ModuleSpec("solana.rpc.api", None)))
    sys.modules.setdefault("solana.rpc.async_api", types.SimpleNamespace(AsyncClient=object, __spec__=importlib.machinery.ModuleSpec("solana.rpc.async_api", None)))
    ws_mod = types.ModuleType("solana.rpc.websocket_api")
    ws_mod.__spec__ = importlib.machinery.ModuleSpec("solana.rpc.websocket_api", None)
    ws_mod.connect = lambda *a, **k: None
    ws_mod.RpcTransactionLogsFilterMentions = object
    sys.modules.setdefault("solana.rpc.websocket_api", ws_mod)

# Additional optional dependency stubs
for _mod_name in ["cachetools", "sqlalchemy", "watchfiles", "psutil", "flask"]:
    if importlib.util.find_spec(_mod_name) is None:
        mod = types.ModuleType(_mod_name)
        mod.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        sys.modules.setdefault(_mod_name, mod)

if "flask" in sys.modules:
    flask_mod = sys.modules["flask"]
    if not hasattr(flask_mod, "Flask"):
        class _DummyResponse(dict):
            status_code = 200

            def get_json(self):
                return dict(self)

        class _DummyApp:
            def __init__(self, *a, **k):
                self._routes = {}

            def route(self, path, methods=None, **_):
                methods = methods or ["GET"]

                def decorator(func):
                    for m in methods:
                        self._routes[(m.upper(), path)] = func
                    return func

                return decorator

            def test_client(self):
                app = self

                class Client:
                    def get(self, path, **_):
                        func = app._routes.get(("GET", path))
                        return _DummyResponse() if func is None else _DummyResponse(**(func() or {}))

                    def post(self, path, json=None, data=None, **_):
                        flask_mod.request.json = json
                        flask_mod.request.form = data if isinstance(data, dict) else {}
                        func = app._routes.get(("POST", path))
                        return _DummyResponse() if func is None else _DummyResponse(**(func() or {}))

                return Client()

        flask_mod.Flask = _DummyApp
        flask_mod.jsonify = lambda obj=None, **kw: obj if obj is not None else kw
        flask_mod.render_template_string = lambda *a, **k: ""
        flask_mod.request = types.SimpleNamespace(json=None, form={}, files={}, args={}, method="GET", get_json=lambda: flask_mod.request.json)

if importlib.util.find_spec("numpy") is None:
    np_mod = types.ModuleType("numpy")
    np_mod.__spec__ = importlib.machinery.ModuleSpec("numpy", None)

    def _listify(obj):
        return list(obj)

    def linspace(start, stop, num=50):
        step = (stop - start) / (num - 1) if num > 1 else 0
        return [start + i * step for i in range(num)]

    def array(obj, dtype=None):
        return _listify(obj)

    asarray = array

    def zeros(n, dtype=None):
        return [0.0] * (n if isinstance(n, int) else n[0] * n[1])

    def ones(n, dtype=None):
        return [1.0] * (n if isinstance(n, int) else n[0] * n[1])

    def full(n, val, dtype=None):
        return [val] * (n if isinstance(n, int) else n[0] * n[1])

    def cumsum(seq):
        out = []
        total = 0
        for x in seq:
            total += x
            out.append(total)
        return out

    class _Random:
        def normal(self, mean, _std, size=None):
            size = size if size is not None else 1
            if isinstance(size, tuple):
                size = size[0]
            return [mean] * size

        def seed(self, _seed):
            pass

    np_mod.random = _Random()

    def array_equal(a, b):
        return list(a) == list(b)

    def searchsorted(a, v, side="left"):
        from bisect import bisect_left, bisect_right

        return (bisect_left if side == "left" else bisect_right)(a, v)

    def argsort(seq):
        return sorted(range(len(seq)), key=lambda i: seq[i])

    def unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _any(seq):
        return any(seq)

    def _all(seq):
        return all(seq)

    def isnan(x):
        import math

        if isinstance(x, list):
            return [math.isnan(v) for v in x]
        return math.isnan(x)

    np_mod.array = array
    np_mod.asarray = asarray
    np_mod.zeros = zeros
    np_mod.ones = ones
    np_mod.full = full
    np_mod.linspace = linspace
    np_mod.cumsum = cumsum
    np_mod.array_equal = array_equal
    np_mod.random = np_mod.random
    np_mod.float32 = float
    np_mod.float64 = float
    np_mod.searchsorted = searchsorted
    np_mod.argsort = argsort
    np_mod.unique = unique
    np_mod.any = _any
    np_mod.all = _all
    np_mod.isnan = isnan
    np_mod.column_stack = lambda tup: [list(x) for x in zip(*tup)]
    np_mod.ndarray = list

    sys.modules.setdefault("numpy", np_mod)

# Ensure project root is in sys.path when running tests directly with 'pytest'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import asyncio

from solhunter_zero.http import close_session


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="slow test: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _close_http_session():
    yield
    asyncio.run(close_session())
