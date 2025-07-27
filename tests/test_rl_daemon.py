import asyncio
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


