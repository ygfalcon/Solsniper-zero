import os
import sys
import types
import importlib.util
import importlib.machinery

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

if importlib.util.find_spec("aiofiles") is None:
    aiof_mod = types.ModuleType("aiofiles")
    aiof_mod.__spec__ = importlib.machinery.ModuleSpec("aiofiles", None)
    sys.modules.setdefault("aiofiles", aiof_mod)


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
if importlib.util.find_spec("numpy") is None:
    np_mod = types.ModuleType("numpy")
    np_mod.__spec__ = importlib.machinery.ModuleSpec("numpy", None)

    def _to_list(obj, fill=0.0):
        if isinstance(obj, (list, tuple)):
            return [list(o) if isinstance(o, (list, tuple)) else o for o in obj]
        return [obj]

    np_mod.array = lambda x, dtype=None: _to_list(x)
    np_mod.asarray = lambda x, dtype=None: _to_list(x)

    def _fill(shape, value):
        if isinstance(shape, int):
            return [value for _ in range(shape)]
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return [[value for _ in range(shape[1])] for _ in range(shape[0])]
        return [value]

    np_mod.zeros = lambda shape, dtype=float: _fill(shape, 0.0)
    np_mod.ones = lambda shape, dtype=float: _fill(shape, 1.0)
    np_mod.full = lambda shape, val, dtype=None: _fill(shape, val)
    sys.modules.setdefault("numpy", np_mod)

if importlib.util.find_spec("cachetools") is None:
    import collections
    import time

    ct_mod = types.ModuleType("cachetools")
    ct_mod.__spec__ = importlib.machinery.ModuleSpec("cachetools", None)

    class LRUCache(collections.OrderedDict):
        def __init__(self, maxsize=128, *a, **kw):
            self.maxsize = maxsize
            super().__init__()

        def __setitem__(self, key, value):
            if key in self:
                del self[key]
            elif len(self) >= self.maxsize:
                self.popitem(last=False)
            super().__setitem__(key, value)

    class TTLCache(LRUCache):
        def __init__(self, maxsize=128, ttl=600):
            super().__init__(maxsize)
            self.ttl = ttl
            self._exp = {}

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._exp[key] = time.time() + self.ttl

        def __getitem__(self, key):
            if key in self._exp and self._exp[key] < time.time():
                super().pop(key, None)
                self._exp.pop(key, None)
                raise KeyError(key)
            return super().__getitem__(key)

        def __contains__(self, key):
            if key in self._exp and self._exp[key] < time.time():
                super().pop(key, None)
                self._exp.pop(key, None)
                return False
            return super().__contains__(key)

    ct_mod.LRUCache = LRUCache
    ct_mod.TTLCache = TTLCache
    sys.modules.setdefault("cachetools", ct_mod)

if importlib.util.find_spec("sqlalchemy") is None:
    sa = types.ModuleType("sqlalchemy")
    sa.__spec__ = importlib.machinery.ModuleSpec("sqlalchemy", None)

    sa.Column = lambda *a, **k: None
    sa.Integer = sa.Float = sa.String = sa.Text = sa.DateTime = object
    sa.ForeignKey = lambda *a, **k: None
    sa.create_engine = lambda *a, **k: None

    class MetaData:
        def create_all(self, *a, **k):
            pass

    sa.MetaData = MetaData

    class Table:
        def __init__(self, *a, **k):
            pass

    sa.Table = Table
    sa.select = lambda *a, **k: None

    ext = types.ModuleType("sqlalchemy.ext")
    async_mod = types.ModuleType("sqlalchemy.ext.asyncio")
    async_mod.create_async_engine = lambda *a, **k: None
    async_mod.async_sessionmaker = lambda *a, **k: None
    async_mod.AsyncSession = object
    ext.asyncio = async_mod
    sa.ext = ext

    orm = types.ModuleType("sqlalchemy.orm")

    def declarative_base(*a, **k):
        return type("Base", (), {"metadata": MetaData()})

    orm.declarative_base = declarative_base

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def add(self, *a, **k):
            pass

        def commit(self):
            pass

        def query(self, *a, **k):
            return []

    orm.sessionmaker = lambda *a, **k: lambda **kw: DummySession()
    sa.orm = orm
    sys.modules.setdefault("sqlalchemy", sa)
    sys.modules.setdefault("sqlalchemy.ext", ext)
    sys.modules.setdefault("sqlalchemy.ext.asyncio", async_mod)
    sys.modules.setdefault("sqlalchemy.orm", orm)

if importlib.util.find_spec("watchfiles") is None:
    watch_mod = types.ModuleType("watchfiles")
    watch_mod.__spec__ = importlib.machinery.ModuleSpec("watchfiles", None)

    async def awatch(*a, **k):
        if False:
            yield None

    watch_mod.awatch = awatch
    sys.modules.setdefault("watchfiles", watch_mod)

if importlib.util.find_spec("psutil") is None:
    ps_mod = types.ModuleType("psutil")
    ps_mod.__spec__ = importlib.machinery.ModuleSpec("psutil", None)
    ps_mod.cpu_percent = lambda *a, **k: 0.0
    ps_mod.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
    sys.modules.setdefault("psutil", ps_mod)

if importlib.util.find_spec("flask") is None:
    flask_mod = types.ModuleType("flask")
    flask_mod.__spec__ = importlib.machinery.ModuleSpec("flask", None)

    class Request:
        def __init__(self, *, method="GET", json=None, form=None, files=None):
            self.method = method
            self._json = json
            self.form = form or {}
            self.files = files or {}

        def get_json(self):
            return self._json

    class Response:
        def __init__(self, data, status=200):
            self._data = data
            self.status_code = status

        def get_json(self):
            return self._data

    def jsonify(obj=None, **kw):
        if obj is None:
            obj = kw
        return Response(obj)

    def render_template_string(tmpl, **ctx):
        return tmpl.format(**ctx)

    class Flask:
        def __init__(self, name, static_folder=None):
            self.routes = {}

        def route(self, path, methods=None):
            methods = methods or ["GET"]

            def decorator(func):
                for m in methods:
                    self.routes[(path, m)] = func
                return func

            return decorator

        def test_client(self):
            app = self

            class Client:
                def _call(self, method, path, json=None, data=None, content_type=None):
                    form = {}
                    files = {}
                    if data and isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, tuple):
                                fobj, fname = v
                                setattr(fobj, "filename", fname)
                                files[k] = fobj
                            else:
                                form[k] = v
                    flask_mod.request = Request(method=method, json=json, form=form, files=files)
                    func = app.routes.get((path, method))
                    if not func:
                        return Response({"message": "not found"}, 404)
                    res = func()
                    if isinstance(res, Response):
                        return res
                    if isinstance(res, tuple):
                        return Response(res[0], res[1])
                    return Response(res)

                def get(self, path):
                    return self._call("GET", path)

                def post(self, path, json=None, data=None, content_type=None):
                    return self._call("POST", path, json=json, data=data, content_type=content_type)

            return Client()

    flask_mod.Flask = Flask
    flask_mod.jsonify = jsonify
    flask_mod.request = Request()
    flask_mod.render_template_string = render_template_string
    sys.modules.setdefault("flask", flask_mod)

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
