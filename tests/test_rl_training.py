import sys
import types
import importlib.util

dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
if importlib.util.find_spec("pytorch_lightning") is None:
    pl = types.ModuleType("pytorch_lightning")
    callbacks = types.SimpleNamespace(Callback=object)
    pl.callbacks = callbacks
    import torch.nn as _nn
    pl.LightningModule = type(
        "LightningModule",
        (_nn.Module,),
        {"save_hyperparameters": lambda self, *a, **k: None},
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

from solhunter_zero.rl_training import RLTraining
from solhunter_zero.offline_data import OfflineData


def test_rl_training_runs(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    data.log_snapshot("tok", 1.0, 1.0, total_depth=1.5, imbalance=0.0, slippage=0.0, volume=0.0)
    data.log_snapshot("tok", 1.1, 1.0, total_depth=1.6, imbalance=0.0, slippage=0.0, volume=0.0)
    data.log_trade("tok", "buy", 1.0, 1.0)
    data.log_trade("tok", "sell", 1.0, 1.1)

    model_path = tmp_path / "ppo_model.pt"
    trainer = RLTraining(db_url=db, model_path=model_path)
    trainer.train()
    assert model_path.exists()
