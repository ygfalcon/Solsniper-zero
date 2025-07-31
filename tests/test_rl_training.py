import sys
import types
import importlib.util
from pathlib import Path

dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
if importlib.util.find_spec("torch") is None:
    torch_mod = types.ModuleType("torch")
    torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch_mod.__path__ = []
    torch_mod.Tensor = object
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch_mod.device = lambda *a, **k: types.SimpleNamespace(type="cpu")
    torch_mod.load = lambda *a, **k: {}
    def _save(obj, path, *a, **k):
        Path(path).touch()
    torch_mod.save = _save
    torch_mod.tensor = lambda *a, **k: object()
    torch_mod.zeros = lambda *a, **k: object()
    torch_mod.long = int
    sys.modules.setdefault("torch", torch_mod)
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__spec__ = importlib.machinery.ModuleSpec("torch.nn", None)
    torch_nn.__path__ = []
    torch_nn.Module = type(
        "Module",
        (),
        {
            "to": lambda self, *a, **k: None,
            "load_state_dict": lambda self, *a, **k: None,
        },
    )
    torch_nn.Sequential = lambda *a, **k: object()
    torch_nn.Linear = lambda *a, **k: object()
    torch_nn.ReLU = lambda *a, **k: object()
    torch_nn.MSELoss = lambda *a, **k: object()
    sys.modules.setdefault("torch.nn", torch_nn)
    torch_opt = types.ModuleType("torch.optim")
    torch_opt.__spec__ = importlib.machinery.ModuleSpec("torch.optim", None)
    torch_opt.__path__ = []
    sys.modules.setdefault("torch.optim", torch_opt)
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.__spec__ = importlib.machinery.ModuleSpec("torch.utils", None)
    torch_utils.__path__ = []
    sys.modules.setdefault("torch.utils", torch_utils)
    tud = types.ModuleType("torch.utils.data")
    tud.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", None)
    sys.modules.setdefault("torch.utils.data", tud)
    tud.Dataset = object
    tud.DataLoader = object
if importlib.util.find_spec("pytorch_lightning") is None:
    pl = types.ModuleType("pytorch_lightning")
    callbacks = types.SimpleNamespace(Callback=object)
    pl.callbacks = callbacks
    import torch.nn as _nn
    pl.LightningModule = type(
        "LightningModule",
        (_nn.Module,),
        {
            "save_hyperparameters": lambda self, *a, **k: None,
            "state_dict": lambda self: {},
        },
    )
    pl.LightningDataModule = type("LightningDataModule", (), {})
    pl.Trainer = type(
        "Trainer",
        (),
        {
            "__init__": lambda self, *a, **k: None,
            "fit": lambda *a, **k: None,
        },
    )
    sys.modules.setdefault("pytorch_lightning", pl)

if importlib.util.find_spec("solders") is None:
    sys.modules.setdefault("solders", types.ModuleType("solders"))
    sys.modules["solders.keypair"] = types.SimpleNamespace(Keypair=type("Keypair", (), {}))
    sys.modules["solders.pubkey"] = types.SimpleNamespace(Pubkey=object)
    sys.modules["solders.hash"] = types.SimpleNamespace(Hash=object)
    sys.modules["solders.message"] = types.SimpleNamespace(MessageV0=object)
    sys.modules["solders.transaction"] = types.SimpleNamespace(VersionedTransaction=object)
if importlib.util.find_spec("solana") is None:
    sys.modules.setdefault("solana", types.ModuleType("solana"))
    sys.modules["solana.rpc"] = types.ModuleType("rpc")
    sys.modules["solana.rpc.api"] = types.SimpleNamespace(Client=object)
    sys.modules["solana.rpc.async_api"] = types.SimpleNamespace(AsyncClient=object)
    sys.modules["solana.rpc.websocket_api"] = types.SimpleNamespace(connect=lambda *a, **k: None)
    sys.modules["solana.rpc.websocket_api"].RpcTransactionLogsFilterMentions = object
try:
    import google
except ModuleNotFoundError:
    google = types.ModuleType("google")
    google.__path__ = []
    sys.modules.setdefault("google", google)
if importlib.util.find_spec("google.protobuf") is None:
    protobuf = types.ModuleType("protobuf")
    descriptor = types.ModuleType("descriptor")
    descriptor_pool = types.ModuleType("descriptor_pool")
    symbol_database = types.ModuleType("symbol_database")
    symbol_database.Default = lambda: object()
    internal = types.ModuleType("internal")
    internal.builder = types.ModuleType("builder")
    protobuf.descriptor = descriptor
    protobuf.descriptor_pool = descriptor_pool
    protobuf.symbol_database = symbol_database
    protobuf.internal = internal
    google.protobuf = protobuf
    sys.modules.setdefault("google.protobuf", protobuf)
    sys.modules.setdefault("google.protobuf.descriptor", descriptor)
    sys.modules.setdefault("google.protobuf.descriptor_pool", descriptor_pool)
    sys.modules.setdefault("google.protobuf.symbol_database", symbol_database)
    sys.modules.setdefault("google.protobuf.internal", internal)
    sys.modules.setdefault("google.protobuf.internal.builder", internal.builder)

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
    "Event",
]:
    setattr(event_pb2, name, object())
sys.modules.setdefault("solhunter_zero.event_pb2", event_pb2)

from solhunter_zero.rl_training import RLTraining
from solhunter_zero.offline_data import OfflineData
from scripts import build_mmap_dataset
import pytest


@pytest.mark.asyncio
async def test_rl_training_runs(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    await data.log_snapshot("tok", 1.0, 1.0, total_depth=1.5, imbalance=0.0, slippage=0.0, volume=0.0)
    await data.log_snapshot("tok", 1.1, 1.0, total_depth=1.6, imbalance=0.0, slippage=0.0, volume=0.0)
    await data.log_trade("tok", "buy", 1.0, 1.0)
    await data.log_trade("tok", "sell", 1.0, 1.1)

    model_path = tmp_path / "ppo_model.pt"
    trainer = RLTraining(db_url=db, model_path=model_path)
    await trainer.train()
    assert model_path.exists()


def test_dynamic_worker_count(monkeypatch, tmp_path):
    import solhunter_zero.rl_training as rl_training

    class DummyDataset(rl_training.Dataset):
        def __len__(self):
            return 400

        def __getitem__(self, idx):
            return (
                rl_training.torch.zeros(9),
                rl_training.torch.tensor(0, dtype=rl_training.torch.long),
                rl_training.torch.tensor(0.0),
            )

    counts = []

    class DummyLoader:
        def __init__(self, *a, **kw):
            counts.append(kw.get("num_workers"))

    monkeypatch.setattr(rl_training, "DataLoader", DummyLoader)
    monkeypatch.setattr(rl_training, "_TradeDataset", lambda *a, **k: DummyDataset())
    monkeypatch.setattr(rl_training.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(rl_training.torch, "save", lambda *a, **k: None)

    rl_training.fit(
        [],
        [],
        model_path=tmp_path / "m.pt",
        dynamic_workers=True,
        cpu_callback=lambda: 80.0,
    )
    high = counts[-1]
    rl_training.fit(
        [],
        [],
        model_path=tmp_path / "m.pt",
        dynamic_workers=True,
        cpu_callback=lambda: 10.0,
    )
    low = counts[-1]
    assert low > high


def test_workers_adjusted_each_epoch(monkeypatch):
    import solhunter_zero.rl_training as rl_training

    class DummyDataset(rl_training.Dataset):
        def __len__(self):
            return 400

        def __getitem__(self, idx):
            return (
                rl_training.torch.zeros(9),
                rl_training.torch.tensor(0, dtype=rl_training.torch.long),
                rl_training.torch.tensor(0.0),
            )

    class DummyLoader:
        def __init__(self, *a, **kw):
            self.num_workers = kw.get("num_workers")
            self.pin_memory = kw.get("pin_memory")
            self.persistent_workers = kw.get("persistent_workers")

    monkeypatch.setattr(rl_training, "DataLoader", lambda *a, **kw: DummyLoader(*a, **kw))
    monkeypatch.setattr(rl_training, "_TradeDataset", lambda *a, **k: DummyDataset())
    monkeypatch.setattr(rl_training.os, "cpu_count", lambda: 4)

    cpu_val = {"v": 80.0}
    dm = rl_training.TradeDataModule("db", dynamic_workers=True, cpu_callback=lambda: cpu_val["v"])
    dm.dataset = DummyDataset()
    loader = dm.train_dataloader()
    cb = rl_training._DynamicWorkersCallback()
    trainer = types.SimpleNamespace(datamodule=dm, train_dataloader=loader)

    cb.on_train_epoch_start(trainer, rl_training.LightningPPO())
    high = loader.num_workers
    cpu_val["v"] = 10.0
    cb.on_train_epoch_start(trainer, rl_training.LightningPPO())
    low = loader.num_workers
    assert low > high


def test_worker_count_reduced_when_memory_high(monkeypatch, tmp_path):
    import solhunter_zero.rl_training as rl_training

    class DummyDataset(rl_training.Dataset):
        def __len__(self):
            return 400

        def __getitem__(self, idx):
            return (
                rl_training.torch.zeros(9),
                rl_training.torch.tensor(0, dtype=rl_training.torch.long),
                rl_training.torch.tensor(0.0),
            )

    counts = []

    class DummyLoader:
        def __init__(self, *a, **kw):
            counts.append(kw.get("num_workers"))

    monkeypatch.setattr(rl_training, "DataLoader", DummyLoader)
    monkeypatch.setattr(rl_training, "_TradeDataset", lambda *a, **k: DummyDataset())
    monkeypatch.setattr(rl_training.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(rl_training.torch, "save", lambda *a, **k: None)

    monkeypatch.setattr(
        rl_training.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(percent=90.0),
    )
    rl_training.fit([], [], model_path=tmp_path / "m.pt")
    high = counts[-1]

    monkeypatch.setattr(
        rl_training.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(percent=50.0),
    )
    rl_training.fit([], [], model_path=tmp_path / "m.pt")
    low = counts[-1]

    assert low > high


@pytest.mark.asyncio
async def test_mmap_preferred_when_available(monkeypatch, tmp_path):
    import numpy as np
    from pathlib import Path
    from solhunter_zero.offline_data import OfflineData

    db_url = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db_url)
    await data.log_snapshot("tok", 1.0, 1.0, total_depth=1.0, imbalance=0.0, slippage=0.0, volume=0.0)
    await data.log_trade("tok", "buy", 1.0, 1.0)

    mmap_dir = tmp_path / "datasets"
    mmap_dir.mkdir()
    mmap_path = mmap_dir / "offline_data.npz"
    await data.export_npz(mmap_path)

    monkeypatch.chdir(tmp_path)

    called = {}
    orig_load = np.load

    def fake_load(path, *a, **kw):
        called["path"] = str(path)
        return orig_load(path, *a, **kw)

    monkeypatch.setattr(np, "load", fake_load)
    monkeypatch.setattr(OfflineData, "list_trades", lambda *a, **k: (_ for _ in ()).throw(AssertionError("db used")))
    monkeypatch.setattr(OfflineData, "list_snapshots", lambda *a, **k: (_ for _ in ()).throw(AssertionError("db used")))

    trainer = RLTraining(db_url=db_url, model_path=tmp_path / "m.pt")
    await trainer.data.setup()

    assert Path(called["path"]).resolve() == mmap_path.resolve()
    assert Path(trainer.data.mmap_path).resolve() == mmap_path.resolve()


@pytest.mark.asyncio
async def test_mmap_generated_when_missing(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db_url)
    await data.log_snapshot("tok", 1.0, 1.0, imbalance=0.0, total_depth=1.0)
    await data.log_trade("tok", "buy", 1.0, 1.0)

    out_dir = tmp_path / "datasets"
    out_dir.mkdir()
    mmap_path = out_dir / "offline_data.npz"

    called = {}

    def fake_main(args=None):
        called["args"] = args
        mmap_path.write_bytes(b"x")
        return 0

    monkeypatch.setattr(build_mmap_dataset, "main", fake_main)
    monkeypatch.chdir(tmp_path)

    trainer = RLTraining(db_url=db_url, model_path=tmp_path / "m.pt")

    assert mmap_path.exists()
    assert Path(trainer.data.mmap_path).resolve() == mmap_path.resolve()
    assert "--db" in called.get("args", [])


