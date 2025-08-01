import json
import pytest
pytest.importorskip("torch.nn.utils.rnn")
from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.conviction import ConvictionAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory


def test_mutations_spawn_and_prune(tmp_path):
    mem = Memory('sqlite:///:memory:')
    mem_agent = MemoryAgent(mem)
    base = ConvictionAgent(threshold=0.1)
    path = tmp_path / 'state.json'
    mgr = AgentManager([base, mem_agent], memory_agent=mem_agent, mutation_path=str(path))

    mgr.spawn_mutations(1)
    active_names = [m if isinstance(m, str) else m.get('name') for m in mgr.mutation_state['active']]
    mutated = [a for a in mgr.agents if a.name != base.name and a.name in active_names]
    assert mutated
    m = mutated[0]

    from types import SimpleNamespace
    trades = [
        SimpleNamespace(direction='buy', amount=1, price=1, reason=m.name),
        SimpleNamespace(direction='sell', amount=1, price=0.5, reason=m.name),
    ]
    mem.list_trades = lambda *a, **k: trades

    mgr.prune_underperforming(0.0)
    assert m.name not in [a.name for a in mgr.agents]

    mgr.save_mutation_state()
    data = json.loads(path.read_text())
    data_active = [m if isinstance(m, str) else m.get('name') for m in data.get('active', [])]
    assert m.name not in data_active


def test_mutation_state_persists(tmp_path):
    mem = Memory('sqlite:///:memory:')
    mem_agent = MemoryAgent(mem)
    base = ConvictionAgent(threshold=0.1)
    path = tmp_path / 'state.json'

    mgr = AgentManager([base, mem_agent], memory_agent=mem_agent, mutation_path=str(path))
    spawned = mgr.spawn_mutations(1)
    assert spawned
    mut_name = spawned[0].name
    mgr.save_mutation_state()

    del mgr

    mgr2 = AgentManager([base, mem_agent], memory_agent=mem_agent, mutation_path=str(path))
    assert any(a.name == mut_name for a in mgr2.agents)
