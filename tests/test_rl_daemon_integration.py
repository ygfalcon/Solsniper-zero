import asyncio
from pathlib import Path

import pytest
import torch
import subprocess

from solhunter_zero.rl_daemon import RLDaemon
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory
from solhunter_zero.offline_data import OfflineData


@pytest.mark.asyncio
async def test_daemon_background_reload(tmp_path, monkeypatch):
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

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    reloaded = asyncio.Event()

    def fake_reload():
        reloaded.set()

    agent = DQNAgent(memory_agent=MemoryAgent(mem), epsilon=0.0, model_path=model_path)
    monkeypatch.setattr(agent, 'reload_weights', fake_reload)

    def fake_train(self):
        Path(self.model_path).write_text('x')
        for ag in self.agents:
            ag.reload_weights()

    monkeypatch.setattr(RLDaemon, 'train', fake_train)

    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=model_path, algo='dqn')
    daemon.register_agent(agent)
    daemon.start(0.01)

    await asyncio.wait_for(reloaded.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_reload_on_timestamp_change(tmp_path, monkeypatch):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    mem.log_trade(token='tok', direction='buy', amount=1, price=1)

    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, imbalance=0.0, total_depth=1.0)

    model_path = tmp_path / 'model.pt'

    reloaded = asyncio.Event()

    def fake_reload():
        reloaded.set()

    agent = DQNAgent(memory_agent=MemoryAgent(mem), epsilon=0.0, model_path=model_path)
    monkeypatch.setattr(agent, 'reload_weights', fake_reload)

    class DummyProc:
        def __init__(self, *a, **k):
            pass
        def poll(self):
            return None

    monkeypatch.setattr(subprocess, 'Popen', DummyProc)

    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=model_path, algo='dqn', agents=[agent])
    daemon.start(0.05, auto_train=True, tune_interval=0.05)

    await asyncio.sleep(0.1)
    from solhunter_zero.rl_daemon import _DQN
    torch.save(_DQN().state_dict(), model_path)

    await asyncio.wait_for(reloaded.wait(), timeout=1.0)
