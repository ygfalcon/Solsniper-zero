import asyncio
import sys
import types
import importlib.util

# Stub heavy optional dependencies similar to event_bus tests
dummy_trans = types.ModuleType("transformers")
dummy_trans.pipeline = lambda *a, **k: lambda x: []
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", dummy_trans)
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
if importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
    sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
if importlib.util.find_spec("numpy") is None:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
if importlib.util.find_spec("aiofiles") is None:
    sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))
if importlib.util.find_spec("sqlalchemy") is None:
    sa = types.ModuleType("sqlalchemy")
    sa.create_engine = lambda *a, **k: None
    sa.MetaData = type("Meta", (), {"create_all": lambda *a, **k: None})
    sa.Column = lambda *a, **k: None
    sa.String = sa.Integer = sa.Float = sa.Numeric = sa.Text = object
    sa.DateTime = object
    sa.ForeignKey = lambda *a, **k: None
    sys.modules.setdefault("sqlalchemy", sa)
    orm = types.ModuleType("orm")

    def declarative_base(*a, **k):
        return type("Base", (), {"metadata": sa.MetaData()})

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
    sys.modules.setdefault("sqlalchemy.orm", orm)
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

from solhunter_zero.rl_daemon import RLDaemon
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory
from solhunter_zero.offline_data import OfflineData
import os
import logging


def _no_load(self):
    self._last_mtime = os.path.getmtime(self.model_path)


def test_rl_daemon_updates_and_agent_reloads(tmp_path, monkeypatch, caplog):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    mem.log_trade(token='tok', direction='buy', amount=1, price=1)
    mem.log_trade(token='tok', direction='sell', amount=1, price=2)

    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, total_depth=1.5, imbalance=0.0)
    data.log_snapshot('tok', 1.1, 1.0, total_depth=1.6, imbalance=0.0)

    model_path = tmp_path / 'model.pt'
    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=model_path, algo='dqn')
    monkeypatch.setattr(DQNAgent, "reload_weights", _no_load)
    agent = DQNAgent(memory_agent=MemoryAgent(mem), epsilon=0.0, model_path=model_path)
    daemon.register_agent(agent)
    first = agent._last_mtime
    with caplog.at_level(logging.INFO):
        daemon.train()
    assert model_path.exists()
    assert agent._last_mtime != first
    assert any("checkpoint" in r.message for r in caplog.records)


def test_queued_trades_trigger_update(tmp_path, monkeypatch):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, imbalance=0.0, total_depth=1.0)

    queue: asyncio.Queue = asyncio.Queue()
    mem_agent = MemoryAgent(mem, offline_data=data, queue=queue)
    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=tmp_path/'model.pt', algo='dqn', queue=queue)

    action = {'token': 'tok', 'side': 'buy', 'amount': 1.0, 'price': 1.0}
    asyncio.run(mem_agent.log(action, skip_db=True))

    daemon.train()
    assert (tmp_path / 'model.pt').exists()


def test_daemon_receives_risk_update(tmp_path):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    data = OfflineData(data_db)

    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=tmp_path/'model.pt')

    from solhunter_zero.event_bus import publish

    publish("risk_updated", {"multiplier": 3.0})
    asyncio.run(asyncio.sleep(0))

    assert daemon.current_risk == 3.0


def test_rl_checkpoint_event_emitted(tmp_path, monkeypatch):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    mem.log_trade(token='tok', direction='buy', amount=1, price=1)

    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, imbalance=0.0, total_depth=1.0)

    events = []
    from solhunter_zero.event_bus import subscribe
    import solhunter_zero.rl_training as rl_training
    from pathlib import Path

    unsub = subscribe("rl_checkpoint", lambda p: events.append(p))

    def fake_fit(*a, **k):
        Path(k.get("model_path")).write_text("x")

    monkeypatch.setattr(rl_training, "fit", fake_fit)
    import torch
    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=tmp_path/'model.pt')
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(daemon.model, "load_state_dict", lambda *_: None)
    daemon.train()
    unsub()

    from solhunter_zero.schemas import RLCheckpoint

    assert events and isinstance(events[0], RLCheckpoint)
    assert events[0].path.endswith("model.pt")


def test_rl_weights_event_emitted(tmp_path, monkeypatch):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    mem.log_trade(token='tok', direction='buy', amount=1, price=1)

    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, imbalance=0.0, total_depth=1.0)

    events = []
    from solhunter_zero.event_bus import subscribe
    import solhunter_zero.rl_training as rl_training
    from pathlib import Path

    unsub = subscribe("rl_weights", lambda p: events.append(p))

    def fake_fit(*a, **k):
        Path(k.get("model_path")).write_text("x")

    monkeypatch.setattr(rl_training, "fit", fake_fit)
    import torch
    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=tmp_path/'model.pt')
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(daemon.model, "load_state_dict", lambda *_: None)
    daemon.train()
    unsub()

    from solhunter_zero.schemas import RLWeights

    assert events and isinstance(events[0], RLWeights)


