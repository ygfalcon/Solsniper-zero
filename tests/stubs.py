# Helper module providing lightweight stubs for optional dependencies.

from __future__ import annotations

import sys
import types
import time
import importlib.machinery


def stub_numpy() -> None:
    if 'numpy' in sys.modules:
        return
    np = types.ModuleType('numpy')
    np.__spec__ = importlib.machinery.ModuleSpec('numpy', None)

    def _make(shape, value):
        if isinstance(shape, int):
            return [value for _ in range(shape)]
        if shape == ():
            return value
        first, *rest = shape
        return [_make(tuple(rest), value) for _ in range(first)]

    def array(obj, dtype=None):
        return list(obj)

    def asarray(obj, dtype=None):
        return list(obj)

    def zeros(shape, dtype=float):
        return _make(shape, 0)

    def ones(shape, dtype=float):
        return _make(shape, 1)

    np.array = array
    np.asarray = asarray
    np.zeros = zeros
    np.ones = ones
    sys.modules.setdefault('numpy', np)


def stub_cachetools() -> None:
    if 'cachetools' in sys.modules:
        return
    ct = types.ModuleType('cachetools')
    ct.__spec__ = importlib.machinery.ModuleSpec('cachetools', None)

    class LRUCache(dict):
        def __init__(self, maxsize: int = 128, *a, **k) -> None:
            self.maxsize = maxsize
            super().__init__(*a, **k)

        def __setitem__(self, key, value) -> None:
            if key in self:
                super().__delitem__(key)
            elif len(self) >= self.maxsize:
                oldest = next(iter(self))
                super().__delitem__(oldest)
            super().__setitem__(key, value)

    class TTLCache(dict):
        def __init__(self, maxsize: int = 128, ttl: float = 60.0) -> None:
            self.maxsize = maxsize
            self.ttl = ttl
            self._exp = {}

        def __setitem__(self, key, value) -> None:
            now = time.monotonic()
            if key in self:
                self._exp[key] = now + self.ttl
                dict.__setitem__(self, key, value)
                return
            while len(self) >= self.maxsize:
                k = next(iter(self))
                if self._exp.get(k, 0) <= now:
                    dict.pop(self, k, None)
                    self._exp.pop(k, None)
                else:
                    break
            self._exp[key] = now + self.ttl
            dict.__setitem__(self, key, value)

        def __getitem__(self, key):
            if key not in self or self._exp.get(key, 0) <= time.monotonic():
                raise KeyError(key)
            return dict.__getitem__(self, key)

        def get(self, key, default=None):
            try:
                return self.__getitem__(key)
            except KeyError:
                return default

        def pop(self, key, default=None):
            self._exp.pop(key, None)
            return dict.pop(self, key, default)

    ct.LRUCache = LRUCache
    ct.TTLCache = TTLCache
    sys.modules.setdefault('cachetools', ct)


def stub_sqlalchemy() -> None:
    if 'sqlalchemy' in sys.modules:
        return
    sa = types.ModuleType('sqlalchemy')
    sa.__spec__ = importlib.machinery.ModuleSpec('sqlalchemy', None)

    class Column:
        def __init__(self, _type=None, *args, **kwargs):
            self.name = None
            self.default = kwargs.get('default')
            self.primary_key = kwargs.get('primary_key', False)

        def __set_name__(self, owner, name):
            self.name = name

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__.get(self.name, self.default)

        def __set__(self, obj, value):
            obj.__dict__[self.name] = value

        def __eq__(self, other):
            return lambda inst: getattr(inst, self.name) == other

        def __gt__(self, other):
            return lambda inst: getattr(inst, self.name) > other

        def contains(self, item):
            return lambda inst: item in getattr(inst, self.name, '')

        def in_(self, seq):
            return lambda inst: getattr(inst, self.name) in seq

    class MetaData:
        def create_all(self, *a, **k):
            pass

    class Table:
        pass

    def create_engine(*a, **k):
        return Engine()

    class Query:
        def __init__(self, model, session=None):
            self.session = session
            self.model = model
            self.filters = []
            self.order = None
            self.limit_n = None

        def with_session(self, session):
            self.session = session
            return self

        def _execute(self):
            if self.session is None:
                raise RuntimeError('unbound query')
            data = list(self.session.engine.storage.get(self.model, []))
            for f in self.filters:
                data = [d for d in data if f(d)]
            if self.order:
                data.sort(key=lambda d: getattr(d, self.order))
            if self.limit_n is not None:
                data = data[:self.limit_n]
            return data

        def __iter__(self):
            return iter(self._execute())

        def all(self):
            return self._execute()

        def filter(self, func):
            self.filters.append(func)
            return self

        def filter_by(self, **kwargs):
            self.filters.append(lambda obj: all(getattr(obj, k) == v for k, v in kwargs.items()))
            return self

        def order_by(self, col):
            self.order = getattr(col, 'name', None)
            return self

        def limit(self, n):
            self.limit_n = n
            return self

    def select(model):
        return Query(model)

    class Engine:
        def __init__(self):
            self.storage = {}
            self.counters = {}

        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def run_sync(self, func):
            func(self)

        async def dispose(self):
            pass

    class AsyncSession:
        def __init__(self, engine):
            self.engine = engine

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def add(self, obj):
            lst = self.engine.storage.setdefault(type(obj), [])
            if getattr(obj, 'id', None) is None:
                idx = self.engine.counters.get(type(obj), 1)
                setattr(obj, 'id', idx)
                self.engine.counters[type(obj)] = idx + 1
            lst.append(obj)

        def bulk_save_objects(self, objs):
            for o in objs:
                self.add(o)

        def commit(self):
            pass

        async def run_sync(self, func):
            func(self)

        async def execute(self, query: Query):
            data = query.with_session(self)._execute()
            return Result(data)

    class Result:
        def __init__(self, data):
            self._data = data

        def scalars(self):
            class _Scalars:
                def __init__(self, data):
                    self._data = data

                def all(self):
                    return self._data

            return _Scalars(self._data)

    def sessionmaker(bind=None, expire_on_commit=False):
        def factory(**kw):
            return Session(bind)
        return factory

    class Session(AsyncSession):
        def query(self, model):
            return Query(model, session=self)

    def declarative_base():
        class Base:
            metadata = MetaData()

            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        return Base

    sa.Column = Column
    sa.Integer = type('Integer', (), {})
    sa.Float = type('Float', (), {})
    sa.String = type('String', (), {})
    sa.DateTime = type('DateTime', (), {})
    sa.Text = type('Text', (), {})
    LargeBinary = object
    sa.LargeBinary = LargeBinary
    sa.ForeignKey = lambda *a, **k: None
    sa.create_engine = create_engine
    sa.MetaData = MetaData
    sa.Table = Table
    sa.select = select

    orm = types.ModuleType('sqlalchemy.orm')
    orm.__spec__ = importlib.machinery.ModuleSpec('sqlalchemy.orm', None)
    orm.sessionmaker = sessionmaker
    orm.declarative_base = declarative_base
    sa.orm = orm

    ext = types.ModuleType('sqlalchemy.ext')
    ext.__spec__ = importlib.machinery.ModuleSpec('sqlalchemy.ext', None)
    async_mod = types.ModuleType('sqlalchemy.ext.asyncio')
    async_mod.__spec__ = importlib.machinery.ModuleSpec('sqlalchemy.ext.asyncio', None)

    def create_async_engine(*a, **k):
        return Engine()

    async_mod.create_async_engine = create_async_engine
    async_mod.async_sessionmaker = sessionmaker
    async_mod.AsyncSession = AsyncSession
    ext.asyncio = async_mod
    sa.ext = ext

    sys.modules.setdefault('sqlalchemy', sa)
    sys.modules.setdefault('sqlalchemy.orm', orm)
    sys.modules.setdefault('sqlalchemy.ext', ext)
    sys.modules.setdefault('sqlalchemy.ext.asyncio', async_mod)


def stub_watchfiles() -> None:
    if 'watchfiles' in sys.modules:
        return
    mod = types.ModuleType('watchfiles')
    mod.__spec__ = importlib.machinery.ModuleSpec('watchfiles', None)

    async def awatch(*a, **k):
        if False:
            yield None

    mod.awatch = awatch
    sys.modules.setdefault('watchfiles', mod)


def stub_psutil() -> None:
    if 'psutil' in sys.modules:
        return
    mod = types.ModuleType('psutil')
    mod.__spec__ = importlib.machinery.ModuleSpec('psutil', None)
    mod.cpu_percent = lambda *a, **k: 0.0
    mod.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
    sys.modules.setdefault('psutil', mod)


def stub_flask() -> None:
    if 'flask' in sys.modules:
        return
    flask = types.ModuleType('flask')
    flask.__spec__ = importlib.machinery.ModuleSpec('flask', None)

    request = types.SimpleNamespace(json=None, files={})

    class Flask:
        def __init__(self, name, static_folder=None):
            self.routes = {}

        def route(self, path, methods=None):
            if methods is None:
                methods = ['GET']
            methods = tuple(m.upper() for m in methods)

            def decorator(func):
                self.routes[(path, methods)] = func
                return func

            return decorator

        def test_client(self):
            app = self

            class Client:
                def open(self, path, method='GET', json=None):
                    request.json = json
                    for (p, m), func in app.routes.items():
                        if p == path and method.upper() in m:
                            data = func()
                            return Response(data)
                    return Response(None)

                def get(self, path):
                    return self.open(path, 'GET')

                def post(self, path, json=None):
                    return self.open(path, 'POST', json=json)

            return Client()

    class Response:
        def __init__(self, data):
            self._data = data

        def get_json(self):
            return self._data

    def jsonify(obj=None):
        return obj if obj is not None else {}

    def render_template_string(tpl):
        return tpl

    flask.Flask = Flask
    flask.jsonify = jsonify
    flask.request = request
    flask.render_template_string = render_template_string
    sys.modules.setdefault('flask', flask)


def stub_requests() -> None:
    if 'requests' in sys.modules:
        return
    mod = types.ModuleType('requests')
    mod.__spec__ = importlib.machinery.ModuleSpec('requests', None)

    class HTTPError(Exception):
        pass

    mod.HTTPError = HTTPError
    mod.get = lambda *a, **k: None
    mod.post = lambda *a, **k: None
    sys.modules.setdefault('requests', mod)


def stub_websockets() -> None:
    if 'websockets' in sys.modules:
        return
    mod = types.ModuleType('websockets')
    mod.__spec__ = importlib.machinery.ModuleSpec('websockets', None)

    class DummyConn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def send(self, msg):
            pass

        async def recv(self):
            await asyncio.sleep(0)
            raise StopAsyncIteration

    async def connect(*a, **k):
        return DummyConn()

    async def serve(handler, host, port):
        class Server:
            sockets = [types.SimpleNamespace(getsockname=lambda: (host, port))]

            def close(self):
                pass

            async def wait_closed(self):
                pass

        return Server()

    import asyncio

    mod.connect = connect
    mod.serve = serve
    sys.modules.setdefault('websockets', mod)


def stub_aiofiles() -> None:
    if 'aiofiles' in sys.modules:
        return
    mod = types.ModuleType('aiofiles')
    mod.__spec__ = importlib.machinery.ModuleSpec('aiofiles', None)

    class _DummyFile:
        def __init__(self):
            self._data = ''

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def read(self):
            return self._data

        async def write(self, data):
            self._data += str(data)
            return len(data)

    def open(*a, **k):
        return _DummyFile()

    mod.open = open
    sys.modules.setdefault('aiofiles', mod)


def stub_bip_utils() -> None:
    if 'bip_utils' in sys.modules:
        return
    mod = types.ModuleType('bip_utils')
    mod.__spec__ = importlib.machinery.ModuleSpec('bip_utils', None)

    class Bip39SeedGenerator:
        def __init__(self, mnemonic: str):
            self.mnemonic = mnemonic

        def Generate(self, passphrase: str = ''):
            return b'\0' * 32

    class _Deriver:
        def Purpose(self):
            return self

        def Coin(self):
            return self

        def Account(self, _):
            return self

        def Change(self, _):
            return self

        def AddressIndex(self, _):
            return self

        def PrivateKey(self):
            return self

        def Raw(self):
            return self

        def ToBytes(self):
            return b'\0' * 32

    class Bip44:
        @staticmethod
        def FromSeed(seed, coin):
            return _Deriver()

    class Bip44Coins:
        SOLANA = object()

    class Bip44Changes:
        CHAIN_EXT = object()

    mod.Bip39SeedGenerator = Bip39SeedGenerator
    mod.Bip44 = Bip44
    mod.Bip44Coins = Bip44Coins
    mod.Bip44Changes = Bip44Changes
    sys.modules.setdefault('bip_utils', mod)


def stub_faiss() -> None:
    if 'faiss' in sys.modules:
        return
    mod = types.ModuleType('faiss')
    mod.__spec__ = importlib.machinery.ModuleSpec('faiss', None)

    class IndexFlatL2:
        def __init__(self, dim):
            self.dim = dim

    class IndexIDMap2:
        def __init__(self, index):
            self.index = index

    mod.IndexFlatL2 = IndexFlatL2
    mod.IndexIDMap2 = IndexIDMap2
    sys.modules.setdefault('faiss', mod)


def stub_sklearn() -> None:
    if 'sklearn' in sys.modules:
        return
    sk = types.ModuleType('sklearn')
    sk.__spec__ = importlib.machinery.ModuleSpec('sklearn', None)
    cluster = types.ModuleType('sklearn.cluster')
    cluster.__spec__ = importlib.machinery.ModuleSpec('sklearn.cluster', None)

    linear_model = types.ModuleType('sklearn.linear_model')
    linear_model.__spec__ = importlib.machinery.ModuleSpec('sklearn.linear_model', None)
    ensemble = types.ModuleType('sklearn.ensemble')
    ensemble.__spec__ = importlib.machinery.ModuleSpec('sklearn.ensemble', None)

    class KMeans:
        def __init__(self, n_clusters=8, n_init='auto'):
            self.n_clusters = n_clusters
            self.cluster_centers_ = [[0.0] for _ in range(n_clusters)]

        def fit_predict(self, X):
            return [0 for _ in range(len(X))]

    class DBSCAN:
        def __init__(self, eps=0.5, min_samples=5):
            pass

        def fit_predict(self, X):
            return [0 for _ in range(len(X))]

    cluster.KMeans = KMeans
    cluster.DBSCAN = DBSCAN
    sk.cluster = cluster

    class LinearRegression:
        pass

    class GradientBoostingRegressor:
        pass

    class RandomForestRegressor:
        pass

    linear_model.LinearRegression = LinearRegression
    ensemble.GradientBoostingRegressor = GradientBoostingRegressor
    ensemble.RandomForestRegressor = RandomForestRegressor
    sk.linear_model = linear_model
    sk.ensemble = ensemble

    sys.modules.setdefault('sklearn', sk)
    sys.modules.setdefault('sklearn.cluster', cluster)
    sys.modules.setdefault('sklearn.linear_model', linear_model)
    sys.modules.setdefault('sklearn.ensemble', ensemble)


def stub_memory() -> None:
    if 'solhunter_zero.memory' in sys.modules:
        return
    mem_mod = types.ModuleType('solhunter_zero.memory')
    mem_mod.__spec__ = importlib.machinery.ModuleSpec('solhunter_zero.memory', None)

    class Trade:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Memory:
        def __init__(self, url: str = 'sqlite:///:memory:') -> None:
            self._trades: list[Trade] = []

        def log_trade(self, **kw) -> int:
            t = Trade(**kw, id=len(self._trades) + 1)
            self._trades.append(t)
            return t.id

        def list_trades(self, *, token: str | None = None, limit: int | None = None, since_id: int | None = None) -> list[Trade]:
            data = [t for t in self._trades if (token is None or t.token == token) and (since_id is None or t.id > since_id)]
            data.sort(key=lambda t: t.id)
            if limit is not None:
                data = data[:limit]
            return data

        def log_var(self, value: float) -> None:
            pass

        def list_vars(self):
            return []

    mem_mod.Memory = Memory
    sys.modules.setdefault('solhunter_zero.memory', mem_mod)


def stub_offline_data() -> None:
    if 'solhunter_zero.offline_data' in sys.modules:
        return
    mod = types.ModuleType('solhunter_zero.offline_data')
    mod.__spec__ = importlib.machinery.ModuleSpec('solhunter_zero.offline_data', None)

    class OfflineData:
        def __init__(self, url: str = 'sqlite:///offline.db') -> None:
            pass

        def log_trade(self, **kw) -> None:
            pass

        async def list_trades(self, *a, **k):
            return []

    mod.OfflineData = OfflineData
    sys.modules.setdefault('solhunter_zero.offline_data', mod)


def stub_torch() -> None:
    """Install minimal torch stubs when torch is unavailable."""
    if 'torch' in sys.modules:
        return

    import contextlib
    import pickle

    torch = types.ModuleType('torch')
    torch.__spec__ = importlib.machinery.ModuleSpec('torch', None)

    torch.manual_seed = lambda *a, **k: None

    class Tensor(list):
        def __init__(self, data=None, **kwargs):
            if isinstance(data, (list, tuple)):
                super().__init__(data)
            elif data is None:
                super().__init__()
            else:
                super().__init__([data])

        def to(self, *a, **k):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return list(self)

        def detach(self):
            return self

        def squeeze(self, *a, **k):
            return self[0] if len(self) == 1 else self

        def unsqueeze(self, *a, **k):
            return Tensor([self])

        def clone(self):
            return Tensor(self)

        def mean(self):
            return sum(self) / len(self) if self else 0.0

    torch.Tensor = Tensor
    torch.tensor = lambda data=None, **k: Tensor(data)
    torch.float32 = object()
    torch.no_grad = contextlib.nullcontext
    class Device:
        def __init__(self, name='cpu'):
            self.type = name

        def __repr__(self):
            return f'device({self.type})'

    torch.device = Device

    def save(obj, path):
        with open(path, 'wb') as fh:
            pickle.dump(obj, fh)

    def load(path, map_location=None):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    torch.save = save
    torch.load = load

    cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.cuda = cuda

    nn = types.ModuleType('torch.nn')
    nn.__spec__ = importlib.machinery.ModuleSpec('torch.nn', None)

    class Module:
        def __init__(self, *a, **k):
            pass

        def parameters(self):
            return []

        def to(self, *a, **k):
            return self

        def load_state_dict(self, *a, **k):
            pass

        def state_dict(self):
            return {}

        def train(self, mode=True):
            return self

        def eval(self):
            return self

        def cpu(self):
            return self

        def __call__(self, *a, **k):
            return Tensor([0.0, 0.0])

    nn.Module = Module

    class Sequential(Module):
        def __init__(self, *mods):
            self.mods = mods

        def __call__(self, x):
            for m in self.mods:
                x = m(x)
            return x

    class Linear(Module):
        def __call__(self, x):
            return Tensor([0.0 for _ in range(2)])

    class ReLU(Module):
        def __call__(self, x):
            return x

    class MSELoss(Module):
        def __call__(self, *a, **k):
            class _Loss:
                def backward(self):
                    pass

            return _Loss()

    nn.Sequential = Sequential
    nn.Linear = Linear
    nn.ReLU = ReLU
    nn.MSELoss = MSELoss

    functional = types.ModuleType('torch.nn.functional')
    functional.__spec__ = importlib.machinery.ModuleSpec('torch.nn.functional', None)
    functional.softmax = lambda x, dim=None: x

    def _mse_loss(*a, **k):
        class _Loss:
            def backward(self):
                pass

        return _Loss()

    functional.mse_loss = _mse_loss
    nn.functional = functional

    nn_utils = types.ModuleType('torch.nn.utils')
    nn_utils.__spec__ = importlib.machinery.ModuleSpec('torch.nn.utils', None)
    rnn_utils = types.ModuleType('torch.nn.utils.rnn')
    rnn_utils.__spec__ = importlib.machinery.ModuleSpec('torch.nn.utils.rnn', None)
    rnn_utils.pad_sequence = lambda seq, batch_first=False, padding_value=0.0: seq
    nn_utils.rnn = rnn_utils
    nn.utils = nn_utils

    torch.nn = nn

    optim = types.ModuleType('torch.optim')
    optim.__spec__ = importlib.machinery.ModuleSpec('torch.optim', None)

    class Adam:
        def __init__(self, params, lr=0.001):
            pass

        def zero_grad(self):
            pass

        def step(self):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, *a, **k):
            pass

    optim.Adam = Adam
    torch.optim = optim

    utils_mod = types.ModuleType('torch.utils')
    utils_mod.__spec__ = importlib.machinery.ModuleSpec('torch.utils', None)

    data_mod = types.ModuleType('torch.utils.data')
    data_mod.__spec__ = importlib.machinery.ModuleSpec('torch.utils.data', None)

    class Dataset:
        pass

    class DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False):
            self.dataset = dataset

        def __iter__(self):
            return iter([])

    data_mod.Dataset = Dataset
    data_mod.DataLoader = DataLoader
    utils_mod.data = data_mod
    torch.utils = utils_mod

    dist_mod = types.ModuleType('torch.distributions')
    dist_mod.__spec__ = importlib.machinery.ModuleSpec('torch.distributions', None)

    class Categorical:
        def __init__(self, logits=None):
            self.logits = logits

        def log_prob(self, actions):
            return Tensor([0.0])

    dist_mod.Categorical = Categorical
    torch.distributions = dist_mod

    sys.modules.setdefault('torch', torch)
    sys.modules.setdefault('torch.nn', nn)
    sys.modules.setdefault('torch.nn.functional', functional)
    sys.modules.setdefault('torch.nn.utils', nn_utils)
    sys.modules.setdefault('torch.nn.utils.rnn', rnn_utils)
    sys.modules.setdefault('torch.optim', optim)
    sys.modules.setdefault('torch.utils', utils_mod)
    sys.modules.setdefault('torch.utils.data', data_mod)
    sys.modules.setdefault('torch.distributions', dist_mod)


def install_stubs() -> None:
    stub_numpy()
    stub_cachetools()
    stub_sqlalchemy()
    stub_watchfiles()
    stub_psutil()
    stub_flask()
    stub_requests()
    stub_aiofiles()
    stub_websockets()
    stub_bip_utils()
    stub_faiss()
    stub_sklearn()
    stub_memory()
    stub_offline_data()
    stub_torch()


install_stubs()
