import asyncio
import sys
import types

# Stub heavy dependencies to avoid installing them
sqlalchemy_mod = types.ModuleType('sqlalchemy')
orm_mod = types.ModuleType('sqlalchemy.orm')
sys.modules['sqlalchemy'] = sqlalchemy_mod
sys.modules['sqlalchemy.orm'] = orm_mod
for name in ['create_engine', 'Column', 'Integer', 'Float', 'String', 'DateTime', 'Text', 'ForeignKey']:
    setattr(sqlalchemy_mod, name, lambda *a, **k: None)
for name in ['declarative_base', 'sessionmaker']:
    setattr(sqlalchemy_mod, name, lambda *a, **k: type('Base', (), {}) if name=='declarative_base' else lambda: None)
    setattr(orm_mod, name, lambda *a, **k: type('Base', (), {}) if name=='declarative_base' else lambda: None)

sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['faiss'] = types.ModuleType('faiss')
sent = types.ModuleType('sentence_transformers')
sent.SentenceTransformer = lambda *a, **k: types.SimpleNamespace(get_sentence_embedding_dimension=lambda:1, encode=lambda x:[0])
sys.modules['sentence_transformers'] = sent
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
sys.modules['requests'] = types.ModuleType('requests')
sklearn = types.ModuleType('sklearn')
sys.modules['sklearn'] = sklearn
sys.modules['sklearn.linear_model'] = types.SimpleNamespace(LinearRegression=object)
sys.modules['sklearn.ensemble'] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
xgb = types.ModuleType('xgboost')
xgb.XGBRegressor = object
sys.modules['xgboost'] = xgb
sys.modules['solders'] = types.ModuleType('solders')
sys.modules['solders.keypair'] = types.ModuleType('keypair')
sys.modules['solders.keypair'].Keypair = object
sys.modules["solders.transaction"] = types.ModuleType("transaction"); sys.modules["solders.transaction"].VersionedTransaction = object
sys.modules["solana"] = types.ModuleType("solana"); sys.modules["solana.rpc"] = types.ModuleType("rpc"); sys.modules["solana.rpc.async_api"] = types.ModuleType("async_api"); sys.modules["solana.rpc.async_api"].AsyncClient = object
sys.modules["solana.rpc.api"] = types.ModuleType("api"); sys.modules["solana.rpc.api"].Client = object

mem_stub = types.ModuleType('solhunter_zero.agents.memory')
mem_stub.MemoryAgent = type('M', (), {'__init__': lambda self, memory=None: None})
sys.modules['solhunter_zero.agents.memory'] = mem_stub

from solhunter_zero.agents.opportunity_cost import OpportunityCostAgent
from solhunter_zero.portfolio import Portfolio, Position
from solhunter_zero.simulation import SimulationResult


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_opportunity_cost_sell_after_flags(monkeypatch):
    pf = DummyPortfolio()
    pf.balances['HOLD'] = Position('HOLD', 1.0, 1.0, 1.0)

    def fake_run(token, count=1):
        roi = 0.1 if token == 'HOLD' else 0.5
        return [SimulationResult(1.0, roi)]

    monkeypatch.setattr(
        'solhunter_zero.agents.opportunity_cost.run_simulations', fake_run
    )

    agent = OpportunityCostAgent(
        candidates=['A', 'B', 'C', 'D', 'E'], memory_agent=None
    )

    actions1 = asyncio.run(agent.propose_trade('HOLD', pf))
    assert actions1 == []

    actions2 = asyncio.run(agent.propose_trade('HOLD', pf))
    assert actions2 and actions2[0]['side'] == 'sell'
