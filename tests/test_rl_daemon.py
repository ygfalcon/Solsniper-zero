from solhunter_zero.rl_daemon import RLDaemon
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory
from solhunter_zero.offline_data import OfflineData
import os


def _no_load(self):
    self._last_mtime = os.path.getmtime(self.model_path)


def test_rl_daemon_updates_and_agent_reloads(tmp_path, monkeypatch):
    mem_db = f"sqlite:///{tmp_path/'mem.db'}"
    data_path = tmp_path / 'data.db'
    data_db = f"sqlite:///{data_path}"

    mem = Memory(mem_db)
    mem.log_trade(token='tok', direction='buy', amount=1, price=1)
    mem.log_trade(token='tok', direction='sell', amount=1, price=2)

    data = OfflineData(data_db)
    data.log_snapshot('tok', 1.0, 1.0, 0.0)
    data.log_snapshot('tok', 1.1, 1.0, 0.0)

    model_path = tmp_path / 'model.pt'
    daemon = RLDaemon(memory_path=mem_db, data_path=str(data_path), model_path=model_path, algo='dqn')
    daemon.train()
    assert model_path.exists()

    monkeypatch.setattr(DQNAgent, "_load_weights", _no_load)
    agent = DQNAgent(memory_agent=MemoryAgent(mem), epsilon=0.0, model_path=model_path)
    first = agent._last_mtime
    daemon.train()
    agent._maybe_reload()
    assert agent._last_mtime != first


