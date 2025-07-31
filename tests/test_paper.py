import asyncio
import sys
import types
import importlib.machinery

_faiss_mod = types.ModuleType('faiss')
_faiss_mod.__spec__ = importlib.machinery.ModuleSpec('faiss', None)
sys.modules.setdefault('faiss', _faiss_mod)
_solders_mod = types.ModuleType('solders')
_solders_mod.__spec__ = importlib.machinery.ModuleSpec('solders', None)
sys.modules.setdefault('solders', _solders_mod)
sys.modules['solders.keypair'] = types.SimpleNamespace(Keypair=type('Keypair', (), {}))
sys.modules['solders.pubkey'] = types.SimpleNamespace(Pubkey=object)
sys.modules['solders.hash'] = types.SimpleNamespace(Hash=object)
sys.modules['solders.message'] = types.SimpleNamespace(MessageV0=object)
sys.modules['solders.transaction'] = types.SimpleNamespace(VersionedTransaction=object)
sys.modules['solders.instruction'] = types.SimpleNamespace(Instruction=object, AccountMeta=object)
sys.modules['solders.signature'] = types.SimpleNamespace(Signature=object)
_aiohttp_mod = types.ModuleType('aiohttp')
_aiohttp_mod.__spec__ = importlib.machinery.ModuleSpec('aiohttp', None)
sys.modules.setdefault('aiohttp', _aiohttp_mod)
_bip_mod = types.ModuleType('bip_utils')
_bip_mod.__spec__ = importlib.machinery.ModuleSpec('bip_utils', None)
sys.modules.setdefault('bip_utils', _bip_mod)
sys.modules['bip_utils'].Bip39SeedGenerator = object
sys.modules['bip_utils'].Bip44 = object
sys.modules['bip_utils'].Bip44Coins = object
sys.modules['bip_utils'].Bip44Changes = object
_sol_mod = types.ModuleType('solana')
_sol_mod.__spec__ = importlib.machinery.ModuleSpec('solana', None)
sys.modules.setdefault('solana', _sol_mod)
sys.modules['solana.rpc'] = types.ModuleType('rpc')
sys.modules['solana.rpc.api'] = types.SimpleNamespace(Client=object)
sys.modules['solana.rpc.commitment'] = types.SimpleNamespace(Confirmed="confirmed")
sys.modules['solana.rpc.async_api'] = types.SimpleNamespace(AsyncClient=object)
sys.modules['solana.rpc.websocket_api'] = types.SimpleNamespace(SolanaWsClient=object)
_np_mod = types.ModuleType('numpy')
_np_mod.__spec__ = importlib.machinery.ModuleSpec('numpy', None)
sys.modules.setdefault('numpy', _np_mod)
import contextlib
torch_mod = types.ModuleType('torch')
torch_mod.__spec__ = importlib.machinery.ModuleSpec('torch', None)
torch_mod.no_grad = contextlib.nullcontext
torch_mod.tensor = lambda *a, **k: None
torch_mod.Tensor = object
torch_mod.nn = types.SimpleNamespace(
    LSTM=object,
    Linear=object,
    TransformerEncoder=object,
    TransformerEncoderLayer=object,
    Module=object,
)
torch_mod.optim = types.ModuleType('optim')
torch_mod.optim.__spec__ = importlib.machinery.ModuleSpec('torch.optim', None)
sys.modules['torch'] = torch_mod
sys.modules['torch.nn'] = torch_mod.nn
sys.modules['torch.optim'] = torch_mod.optim
_st_mod = types.ModuleType('sentence_transformers')
_st_mod.__spec__ = importlib.machinery.ModuleSpec('sentence_transformers', None)
sys.modules.setdefault('sentence_transformers', _st_mod)
memory_mod = types.ModuleType('solhunter_zero.memory')
class DummyMemory:
    def __init__(self, url='sqlite:///:memory:'):
        self.trades = []

    def log_trade(self, **kw):
        self.trades.append(types.SimpleNamespace(**kw))

    def list_trades(self):
        return self.trades

    def log_var(self, value: float) -> None:
        pass

    def list_vars(self):
        return []

memory_mod.Memory = DummyMemory
sys.modules['solhunter_zero.memory'] = memory_mod

portfolio_mod = types.ModuleType('solhunter_zero.portfolio')
class DummyPortfolio:
    def __init__(self, path=None):
        self.path = path
        self.balances = {}
        self.risk_metrics = {}
        self.price_history = {}

    def add(self, token, amount, price):
        self.update(token, amount, price)

    def update(self, *a, **k):
        token, amount, price = a[0], a[1], a[2]
        pos = self.balances.get(token)
        if pos is None:
            if amount > 0:
                self.balances[token] = types.SimpleNamespace(amount=amount, entry_price=price, high_price=price)
        else:
            new_amount = pos.amount + amount
            if new_amount <= 0:
                self.balances.pop(token, None)
            else:
                pos.amount = new_amount

    def total_value(self, prices=None):
        total = 0.0
        for t, pos in self.balances.items():
            price = prices.get(t, 1.0) if prices else 1.0
            total += pos.amount * price
        return total

    def percent_allocated(self, token, prices=None):
        tv = self.total_value(prices)
        if tv == 0:
            return 0.0
        price = prices.get(token, 1.0) if prices else 1.0
        amt = self.balances.get(token, types.SimpleNamespace(amount=0)).amount
        return (amt * price) / tv

    def update_drawdown(self, prices):
        pass

    def record_prices(self, prices):
        pass

    def update_risk_metrics(self):
        pass

    def current_drawdown(self, prices):
        return 0.0

portfolio_mod.Portfolio = DummyPortfolio
def calculate_order_size(*a, **k):
    return 1.0

portfolio_mod.calculate_order_size = calculate_order_size
sys.modules['solhunter_zero.portfolio'] = portfolio_mod
agent_manager_mod = types.ModuleType('solhunter_zero.agent_manager')
class DummyAgentManager:
    def __init__(self, *a, **k):
        pass
    async def execute(self, token, portfolio):
        pass
    def evolve(self, threshold=0.0):
        pass

agent_manager_mod.AgentManager = DummyAgentManager
sys.modules['solhunter_zero.agent_manager'] = agent_manager_mod

advanced_memory_mod = types.ModuleType('solhunter_zero.advanced_memory')
advanced_memory_mod.AdvancedMemory = object
sys.modules['solhunter_zero.advanced_memory'] = advanced_memory_mod
discovery_mod = types.ModuleType('solhunter_zero.agents.discovery')
class StubDiscoveryAgent:
    async def discover_tokens(self, **kwargs):
        return ['TOK']

discovery_mod.DiscoveryAgent = StubDiscoveryAgent
sys.modules['solhunter_zero.agents.discovery'] = discovery_mod
sys.modules['sentence_transformers'].SentenceTransformer = lambda *a, **k: types.SimpleNamespace(
    get_sentence_embedding_dimension=lambda: 1,
    encode=lambda x: [0],
)
sklearn = types.ModuleType('sklearn')
sys.modules.setdefault('sklearn', sklearn)
sys.modules['sklearn.linear_model'] = types.SimpleNamespace(LinearRegression=object)
sys.modules['sklearn.ensemble'] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
sys.modules['xgboost'] = types.ModuleType('xgboost')
sys.modules['xgboost'].XGBRegressor = object

from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult
from solhunter_zero.trade_analyzer import TradeAnalyzer


def test_paper(monkeypatch):
    mem = main_module.Memory('sqlite:///:memory:')
    pf = main_module.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ['TOK']

    monkeypatch.setattr(main_module.DiscoveryAgent, 'discover_tokens', fake_discover)
    monkeypatch.setattr(
        main_module,
        'run_simulations',
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
    )
    monkeypatch.setattr(main_module, 'should_buy', lambda sims: True)
    monkeypatch.setattr(main_module, 'should_sell', lambda sims, **k: False)

    async def fake_place_order(token, side, amount, price, **_):
        return {'order_id': '1'}

    monkeypatch.setattr(main_module, 'place_order_async', fake_place_order)
    import solhunter_zero.gas as gas_mod
    monkeypatch.setattr(gas_mod, 'get_current_fee', lambda testnet=False: 0.0)

    async def _no_fee_async(*_a, **_k):
        return 0.0

    monkeypatch.setattr(gas_mod, 'get_current_fee_async', _no_fee_async)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    trades = mem.list_trades()
    assert len(trades) == 1
    assert trades[0].token == 'TOK'
    assert pf.balances['TOK'].amount > 0


def test_paper_metrics(monkeypatch):
    mem = main_module.Memory('sqlite:///:memory:')
    pf = main_module.Portfolio(path=None)

    async def fake_discover(self, **kwargs):
        return ['TOK']

    monkeypatch.setattr(main_module.DiscoveryAgent, 'discover_tokens', fake_discover)
    monkeypatch.setattr(
        main_module,
        'run_simulations',
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=10.0, liquidity=10.0)],
    )

    def patched_log_trade(self, **kw):
        if kw.get('direction') == 'buy':
            kw['price'] = 1.0
        elif kw.get('direction') == 'sell':
            kw['price'] = 2.0
        kw.setdefault('reason', '')
        self.trades.append(types.SimpleNamespace(**kw))

    monkeypatch.setattr(main_module.Memory, 'log_trade', patched_log_trade)

    async def fake_place_order(token, side, amount, price, **_):
        return {'order_id': '1'}

    monkeypatch.setattr(main_module, 'place_order_async', fake_place_order)
    import solhunter_zero.gas as gas_mod
    monkeypatch.setattr(gas_mod, 'get_current_fee', lambda testnet=False: 0.0)

    async def _no_fee_async(*_a, **_k):
        return 0.0

    monkeypatch.setattr(gas_mod, 'get_current_fee_async', _no_fee_async)

    monkeypatch.setattr(main_module, 'should_buy', lambda sims: True)
    monkeypatch.setattr(main_module, 'should_sell', lambda sims, **k: False)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    monkeypatch.setattr(main_module, 'should_buy', lambda sims: False)
    monkeypatch.setattr(main_module, 'should_sell', lambda sims, **k: True)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=False, offline=True))

    roi = TradeAnalyzer(mem).roi_by_agent()
    assert roi.get('', 0.0) == 1.0
