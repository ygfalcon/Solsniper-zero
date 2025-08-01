import json
import sys
import types
import importlib.machinery

if 'transformers' not in sys.modules:
    trans_mod = types.ModuleType('transformers')
    trans_mod.__spec__ = importlib.machinery.ModuleSpec('transformers', None)
    trans_mod.pipeline = lambda *a, **k: lambda *a2, **k2: None
    sys.modules['transformers'] = trans_mod
else:
    if not hasattr(sys.modules['transformers'], 'pipeline'):
        sys.modules['transformers'].pipeline = lambda *a, **k: lambda *a2, **k2: None

if 'torch' in sys.modules:
    tmod = sys.modules['torch']
else:
    tmod = types.ModuleType('torch')
    tmod.__spec__ = importlib.machinery.ModuleSpec('torch', None)
    sys.modules['torch'] = tmod

class Device:
    def __init__(self, name: str = 'cpu') -> None:
        self.type = name

tmod.device = Device

if not hasattr(tmod, 'nn'):
    nn_mod = types.ModuleType('torch.nn')
    nn_mod.__spec__ = importlib.machinery.ModuleSpec('torch.nn', None)
    tmod.nn = nn_mod
    sys.modules.setdefault('torch.nn', nn_mod)
else:
    nn_mod = tmod.nn

if not hasattr(nn_mod, 'utils'):
    utils_mod = types.ModuleType('torch.nn.utils')
    utils_mod.__spec__ = importlib.machinery.ModuleSpec('torch.nn.utils', None)
    nn_mod.utils = utils_mod
    sys.modules.setdefault('torch.nn.utils', utils_mod)
else:
    utils_mod = nn_mod.utils

if 'torch.nn.utils' not in sys.modules:
    sys.modules['torch.nn.utils'] = utils_mod

if not hasattr(utils_mod, 'rnn'):
    rnn_mod = types.ModuleType('torch.nn.utils.rnn')
    rnn_mod.__spec__ = importlib.machinery.ModuleSpec('torch.nn.utils.rnn', None)
    rnn_mod.pad_sequence = lambda seq, *a, **k: seq
    utils_mod.rnn = rnn_mod
    sys.modules.setdefault('torch.nn.utils.rnn', rnn_mod)

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
    mutated = [a for a in mgr.agents if a.name != base.name and a.name in mgr.mutation_state['active']]
    assert mutated
    m = mutated[0]

    mem.log_trade(token='tok', direction='buy', amount=1, price=1, reason=m.name)
    mem.log_trade(token='tok', direction='sell', amount=1, price=0.5, reason=m.name)
    mem.list_trades = lambda limit=1000: [
        types.SimpleNamespace(direction='buy', amount=1, price=1, reason=m.name),
        types.SimpleNamespace(direction='sell', amount=1, price=0.5, reason=m.name),
    ]

    mgr.prune_underperforming(0.0)
    assert m.name not in [a.name for a in mgr.agents]

    mgr.save_mutation_state()
    data = json.loads(path.read_text())
    assert m.name not in data.get('active', [])


def test_mutation_state_reload(tmp_path):
    mem = Memory('sqlite:///:memory:')
    mem_agent = MemoryAgent(mem)
    base = ConvictionAgent(threshold=0.1)
    path = tmp_path / 'state.json'

    mgr = AgentManager([base, mem_agent], memory_agent=mem_agent, mutation_path=str(path))
    mgr.spawn_mutations(2)
    mgr.save_mutation_state()

    stored = json.loads(path.read_text()).get('active', [])
    mgr2 = AgentManager([base, mem_agent], memory_agent=mem_agent, mutation_path=str(path))

    for name in stored:
        assert name in mgr2.mutation_state.get('active', [])
