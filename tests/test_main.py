import pytest

pytest.importorskip("torch.nn.utils.rnn")
pytest.importorskip("transformers")
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

from solhunter_zero import main as main_module  # noqa: E402
from solhunter_zero.simulation import SimulationResult  # noqa: E402

pytest.importorskip("solders")
from solders.keypair import Keypair  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_arbitrage(monkeypatch):
    async def _fake(*_a, **_k):
        return None

    monkeypatch.setattr(
        main_module.arbitrage, "detect_and_execute_arbitrage", _fake
    )


def test_main_invokes_place_order(monkeypatch):
    async def fake_discover_tokens(self, *_a, **_k):
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "buy", "amount": 1, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)


    called = {}

    async def fake_place_order(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)
    monkeypatch.setattr(main_module.asyncio, "sleep", lambda *_a, **_k: None)

    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=0,
        dry_run=True,
        iterations=1,
    )

    assert called["args"][-2] is True
    assert called["args"][0] == "tok"



# codex/add-offline-option-to-solhunter_zero.main


def test_main_offline(monkeypatch):
    recorded = {}

    async def fake_discover_tokens(self, *, offline=False, token_file=None, method=None):
        recorded["offline"] = offline
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "buy", "amount": 1, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)

    async def fake_place_order_async(*a, **k):
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    async def fake_sleep(_):
        raise SystemExit()

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(SystemExit):
        main_module.main(memory_path="sqlite:///:memory:", loop_delay=0, dry_run=True, offline=True)

    assert recorded["offline"] is True


def test_run_iteration_sells(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("tok", 2, 1.0)
    mem = main_module.Memory("sqlite:///:memory:")


    async def fake_discover_tokens(self, *_a, **_k):
        return []

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "sell", "amount": portfolio.balances[token].amount, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


def test_run_iteration_stop_loss(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("tok", 1, 10.0)
    mem = main_module.Memory("sqlite:///:memory:")


    async def fake_discover_tokens(self, *_a, **_k):
        return []

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "sell", "amount": portfolio.balances[token].amount, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True, stop_loss=0.1))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


def test_run_iteration_take_profit(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("tok", 1, 10.0)
    mem = main_module.Memory("sqlite:///:memory:")


    async def fake_discover_tokens(self, *_a, **_k):
        return []

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return [{"token": token, "side": "sell", "amount": portfolio.balances[token].amount, "price": 0}]

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True, take_profit=0.1))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


@pytest.mark.parametrize(
    "method, target",
    [
        ("onchain", "solhunter_zero.scanner_onchain.scan_tokens_onchain"),
        ("websocket", "solhunter_zero.websocket_scanner.stream_new_tokens"),
        ("mempool", "solhunter_zero.mempool_scanner.stream_mempool_tokens"),
        ("pools", "solhunter_zero.dex_scanner.scan_new_pools"),
        ("file", "solhunter_zero.scanner_common.scan_tokens_from_file"),
    ],
)
def test_discovery_methods(monkeypatch, method, target):
    called = {}

    import solhunter_zero.scanner_common as scanner_common  # noqa: E402
    scanner_common.BIRDEYE_API_KEY = None
    scanner_common.HEADERS.clear()

    async def fake_async(*_a, **_k):
        called["called"] = True
        return []

    async def fake_gen(*_a, **_k):
        called["called"] = True
        if False:
            yield None

    def fake_sync(*_a, **_k):
        called["called"] = True
        return []

    if target.endswith("stream_new_tokens") or target.endswith("stream_mempool_tokens"):
        monkeypatch.setattr(target, fake_gen)
    elif "async" in target or target.endswith("scan_tokens_onchain") or target.endswith("scan_new_pools"):
        monkeypatch.setattr(target, fake_async)
    else:
        monkeypatch.setattr(target, fake_sync)

    import solhunter_zero.token_scanner as async_scanner_mod  # noqa: E402
    async def fake_trend():
        return []
    monkeypatch.setattr(async_scanner_mod, "fetch_trending_tokens_async", fake_trend)
    monkeypatch.setattr(async_scanner_mod, "fetch_raydium_listings_async", fake_trend)
    monkeypatch.setattr(async_scanner_mod, "fetch_orca_listings_async", fake_trend)

    method_name = method

    async def fake_discover_tokens(self, *, offline=False, token_file=None, method=None):
        return await async_scanner_mod.scan_tokens_async(
            offline=offline,
            token_file=token_file,
            method=method or method_name,
        )

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return []

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)
    monkeypatch.setattr(main_module, "place_order_async", lambda *a, **k: None)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)
    monkeypatch.setattr(main_module.asyncio, "sleep", lambda *_a, **_k: None)

    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=0,
        dry_run=True,
        iterations=1,
        discovery_method=method,
    )

    assert called.get("called") is True


def test_run_iteration_arbitrage(monkeypatch):
    pf = main_module.Portfolio(path=None)
    mem = main_module.Memory("sqlite:///:memory:")

    async def fake_discover_tokens(self, *_a, **_k):
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )
    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    called = {}

    async def fake_arbitrage(token, threshold, amount, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, threshold, amount, dry_run)
        return None

    class DummySM:
        def __init__(self, *a, **k):
            pass

        async def evaluate(self, token, portfolio):
            return []

    monkeypatch.setattr(main_module, "StrategyManager", DummySM)
    monkeypatch.setattr(main_module.arbitrage, "detect_and_execute_arbitrage", fake_arbitrage)

    asyncio.run(
        main_module._run_iteration(
            mem,
            pf,
            dry_run=True,
            arbitrage_threshold=0.1,
            arbitrage_amount=2.0,
        )
    )

    assert called["args"] == ("tok", 0.1, 2.0, True)


def test_trade_size_scales_with_portfolio_value(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("hold", 1, 1.0)
    mem = main_module.Memory("sqlite:///:memory:")

    async def fake_discover_tokens(self, *_a, **_k):
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=0, liquidity=0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)

    async def fake_prices(tokens):
        return {t: 1.0 for t in tokens}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_prices)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    called = []

    async def fake_place_order_async(token, side, amount, price, **_):
        called.append(amount)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    monkeypatch.setattr(pf, "total_value", lambda prices=None: 10.0)
    monkeypatch.setattr(pf, "percent_allocated", lambda t, prices=None: 0.0)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))
    size1 = called[-1]

    monkeypatch.setattr(pf, "total_value", lambda prices=None: 20.0)
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))
    size2 = called[-1]

    assert size2 > size1


def test_trade_size_scales_with_risk(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("hold", 1, 1.0)
    mem = main_module.Memory("sqlite:///:memory:")

    async def fake_discover_tokens(self, *_a, **_k):
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(1.0, 1.0, volume=0, liquidity=0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "should_sell", lambda sims, **k: False)

    async def fake_prices(tokens):
        return {t: 1.0 for t in tokens}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_prices)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    called = []

    async def fake_place_order_async(token, side, amount, price, **_):
        called.append(amount)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)

    monkeypatch.setattr(pf, "total_value", lambda prices=None: 10.0)
    monkeypatch.setattr(pf, "percent_allocated", lambda t, prices=None: 0.0)
    monkeypatch.setenv("RISK_MULTIPLIER", "1.0")
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))
    size1 = called[-1]

    monkeypatch.setenv("RISK_MULTIPLIER", "2.0")
    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))
    size2 = called[-1]

    assert size2 > size1


def test_flashloan_amount_scales_with_portfolio(monkeypatch):
    pf = main_module.Portfolio(path=None)
    mem = main_module.Memory("sqlite:///:memory:")

    async def fake_discover_tokens(self, *_a, **_k):
        return ["tok"]

    monkeypatch.setattr(
        main_module.DiscoveryAgent, "discover_tokens", fake_discover_tokens
    )
    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    borrowed = []

    async def fake_borrow(amount, token, inst, *, payer):
        borrowed.append(amount)
        return "sig"

    async def fake_repay(_):
        return True

    monkeypatch.setattr(main_module.arbitrage, "borrow_flash", fake_borrow)
    monkeypatch.setattr(main_module.arbitrage, "repay_flash", fake_repay)

    monkeypatch.setenv("FLASH_LOAN_RATIO", "0.5")

    monkeypatch.setattr(pf, "total_value", lambda prices=None: 10.0)
    asyncio.run(
        main_module._run_iteration(
            mem,
            pf,
            dry_run=True,
            arbitrage_threshold=0.1,
            arbitrage_amount=100.0,
        )
    )
    amt1 = borrowed[-1]

    borrowed.clear()
    monkeypatch.setattr(pf, "total_value", lambda prices=None: 20.0)
    asyncio.run(
        main_module._run_iteration(
            mem,
            pf,
            dry_run=True,
            arbitrage_threshold=0.1,
            arbitrage_amount=100.0,
        )
    )
    amt2 = borrowed[-1]

    assert amt2 > amt1


def test_run_auto_uses_highrisk_and_selects_key(monkeypatch, tmp_path):
    import solhunter_zero.config as cfg_mod  # noqa: E402
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(cfg_mod, "ACTIVE_CONFIG_FILE", str(tmp_path / "cfg" / "active"))
    monkeypatch.setattr(main_module, "CONFIG_DIR", str(tmp_path / "cfg"))
    os.makedirs(cfg_mod.CONFIG_DIR, exist_ok=True)

    keys_dir = tmp_path / "keys"
    monkeypatch.setattr(main_module.wallet, "KEYPAIR_DIR", str(keys_dir))
    monkeypatch.setattr(main_module.wallet, "ACTIVE_KEYPAIR_FILE", str(keys_dir / "active"))
    os.makedirs(keys_dir, exist_ok=True)

    kp = Keypair()
    (keys_dir / "only.json").write_text(json.dumps(list(kp.to_bytes())))

    monkeypatch.delenv("KEYPAIR_PATH", raising=False)

    called = {}

    def fake_main(**kwargs):
        called["path"] = kwargs.get("config_path")

    monkeypatch.setattr(main_module, "main", fake_main)
    async def fake_sync():
        called["sync"] = True
    import sys  # noqa: E402
    import types  # noqa: E402
    sys.modules["solhunter_zero.data_sync"] = types.SimpleNamespace(sync_recent=fake_sync)
    if hasattr(main_module, "data_sync"):
        main_module.data_sync = sys.modules["solhunter_zero.data_sync"]

    main_module.run_auto()

    assert called["path"].endswith("config.highrisk.toml")
    assert (keys_dir / "active").read_text() == "only"
    assert os.getenv("KEYPAIR_PATH") == str(keys_dir / "only.json")


def test_run_auto_uses_selected_config(monkeypatch, tmp_path):
    cfg_dir = tmp_path / "cfg"
    import solhunter_zero.config as cfg_mod  # noqa: E402
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", str(cfg_dir))
    monkeypatch.setattr(cfg_mod, "ACTIVE_CONFIG_FILE", str(cfg_dir / "active"))
    monkeypatch.setattr(main_module, "CONFIG_DIR", str(cfg_dir))
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_file = cfg_dir / "my.toml"
    cfg_file.write_text("risk_tolerance=0.5")
    (cfg_dir / "active").write_text("my.toml")

    called = {}

    def fake_main(**kwargs):
        called["path"] = kwargs.get("config_path")

    monkeypatch.setattr(main_module, "main", fake_main)
    async def fake_sync():
        called["sync"] = True
    import sys  # noqa: E402
    import types  # noqa: E402
    sys.modules["solhunter_zero.data_sync"] = types.SimpleNamespace(sync_recent=fake_sync)
    if hasattr(main_module, "data_sync"):
        main_module.data_sync = sys.modules["solhunter_zero.data_sync"]

    main_module.run_auto()

    assert called["path"] == str(cfg_file)


def test_run_iteration_agent_manager(monkeypatch):
    pf = main_module.Portfolio(path=None)
    mem = main_module.Memory("sqlite:///:memory:")

    async def fake_scan_tokens_async(*, offline=False, token_file=None, method="websocket"):
        return ["tok"]

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)

    called = {}

    class DummyManager:
        async def execute(self, token, portfolio):
            called["token"] = token

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True, agent_manager=DummyManager()))

    assert called.get("token") == "tok"


def test_main_uses_agent_manager(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('agents=["dummy"]')

    executed = []

    class DummyManager:
        def __init__(self):
            self.tokens = []

        @classmethod
        def from_config(cls, cfg):
            executed.append("cfg")
            return cls()

        async def execute(self, token, portfolio):
            executed.append(token)

    monkeypatch.setattr(main_module, "AgentManager", DummyManager)

    def fail(*_a, **_k):
        raise AssertionError("StrategyManager used")

    monkeypatch.setattr(main_module, "StrategyManager", fail)

    async def fake_scan(**k):
        return ["tok"]

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan)
    monkeypatch.setattr(main_module, "place_order_async", lambda *a, **k: None)

    async def fake_log_trade(*a, **k):
        pass

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)
    monkeypatch.setattr(main_module.asyncio, "sleep", lambda *_a, **_k: None)

    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=0,
        dry_run=True,
        iterations=1,
        config_path=str(cfg_file),
    )

    assert executed == ["cfg", "tok"]


def test_scheduler_adjusts_delay(monkeypatch):
    metrics = [
        {"liquidity": 1.0, "volume": 1.0},
        {"liquidity": 50.0, "volume": 50.0},
        {"liquidity": 1.0, "volume": 1.0},
        {"liquidity": 1.0, "volume": 1.0},
    ]

    async def fake_run_iteration(*_a, **_k):
        main_module._LAST_TOKENS = ["tok"]

    monkeypatch.setattr(main_module, "_run_iteration", fake_run_iteration)
    async def fake_fetch(t, base_url=None):
        return metrics.pop(0)

    monkeypatch.setattr(
        main_module,
        "fetch_dex_metrics_async",
        fake_fetch,
    )

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=4,
        iterations=4,
        min_delay=1,
        max_delay=8,
    )

    assert sleeps == [4, 2, 4]


def test_delay_adjusts_with_cpu_and_updates(monkeypatch):
    metrics = [
        {"liquidity": 1.0, "volume": 1.0},
        {"liquidity": 50.0, "volume": 50.0},
        {"liquidity": 1.0, "volume": 1.0},
        {"liquidity": 1.0, "volume": 1.0},
    ]
    cpu = [10.0, 30.0, 10.0]
    updates = [0, 12, 0]
    handlers = []

    def fake_subscribe(topic, handler):
        handlers.append(handler)
        return lambda: None

    monkeypatch.setattr(main_module.event_bus, "subscribe", fake_subscribe)

    async def fake_run_iteration(*_a, **_k):
        main_module._LAST_TOKENS = ["tok"]
        h = handlers[0]
        for _ in range(updates.pop(0)):
            h(None)

    monkeypatch.setattr(main_module, "_run_iteration", fake_run_iteration)

    async def fake_fetch(_t, base_url=None):
        return metrics.pop(0)

    monkeypatch.setattr(main_module, "fetch_dex_metrics_async", fake_fetch)

    def fake_cpu():
        return cpu.pop(0)

    monkeypatch.setattr(main_module.psutil, "cpu_percent", fake_cpu)

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    main_module.arbitrage.DEPTH_RATE_LIMIT = 0.1
    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=4,
        iterations=4,
        min_delay=1,
        max_delay=8,
        depth_freq_low=1.0,
        depth_freq_high=10.0,
        cpu_low_threshold=20.0,
        cpu_high_threshold=80.0,
    )

    assert sleeps == [4, 2, 4]
    assert main_module.arbitrage.DEPTH_RATE_LIMIT == 0.1


def test_agent_manager_evolves(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('agents=["dummy"]')

    calls = []

    class DummyManager:
        evolve_interval = 2
        mutation_threshold = 0.0

        def __init__(self):
            pass

        @classmethod
        def from_config(cls, cfg):
            return cls()

        async def execute(self, token, portfolio):
            pass

        def update_weights(self):
            pass

        def save_weights(self):
            pass

        def evolve(self, *_, **__):
            calls.append(True)

    async def fake_run_iteration(*_a, **_k):
        main_module._LAST_TOKENS = []

    monkeypatch.setattr(main_module, "AgentManager", DummyManager)
    monkeypatch.setattr(main_module, "StrategyManager", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "_run_iteration", fake_run_iteration)
    async def fake_fetch(*_a, **_k):
        return {}

    monkeypatch.setattr(main_module, "fetch_dex_metrics_async", fake_fetch)

    async def fake_sleep(_):
        pass

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    main_module.main(
        memory_path="sqlite:///:memory:",
        loop_delay=0,
        iterations=4,
        config_path=str(cfg_file),
        dry_run=True,
    )

    assert len(calls) == 2
