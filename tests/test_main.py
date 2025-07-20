import pytest
from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult
import asyncio


def test_main_invokes_place_order(monkeypatch):
    # prepare mocks

    async def fake_scan_tokens_async(offline=False, token_file=None):

        return ["tok"]

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(success_prob=1.0, expected_roi=2.0, volume=200.0, liquidity=400.0)
        ],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    async def fake_prices(tokens):
        return {t: 1.0 for t in tokens}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_prices)

    called = {}

    async def fake_place_order(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}
    monkeypatch.setattr(main_module, "place_order_async", fake_place_order)

    # avoid DB writes
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    monkeypatch.setattr(main_module.asyncio, "sleep", lambda *_args, **_kw: None)

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


    async def fake_scan_tokens_async(*, offline=False, token_file=None):

        recorded["offline"] = offline
        return ["tok"]

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(success_prob=1.0, expected_roi=2.0, volume=200.0, liquidity=400.0)
        ],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    async def fake_place_order_async(*a, **k):
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)
    async def fake_prices(tokens):
        return {t: 1.0 for t in tokens}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_prices)
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
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


    async def fake_scan_tokens_async(*, offline=False, token_file=None):

        return []

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(0.2, -0.1, volume=200.0, liquidity=400.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims: True)
    async def fake_prices(tokens):
        return {t: 1.0 for t in tokens}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_prices)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


def test_run_iteration_stop_loss(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("tok", 1, 10.0)
    mem = main_module.Memory("sqlite:///:memory:")


    async def fake_scan_tokens_async(*, offline=False, token_file=None):

        return []

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(0.9, 0.2, volume=200.0, liquidity=400.0)
        ],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims: False)

    async def fake_fetch_token_prices_async(tokens):
        return {"tok": 8.0}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_fetch_token_prices_async)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True, stop_loss=0.1))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


def test_run_iteration_take_profit(monkeypatch):
    pf = main_module.Portfolio(path=None)
    pf.add("tok", 1, 10.0)
    mem = main_module.Memory("sqlite:///:memory:")


    async def fake_scan_tokens_async(*, offline=False, token_file=None):

        return []

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [
            SimulationResult(0.9, 0.2, volume=200.0, liquidity=400.0)
        ],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims: False)

    async def fake_fetch_token_prices_async(tokens):
        return {"tok": 12.0}

    monkeypatch.setattr(main_module, "fetch_token_prices_async", fake_fetch_token_prices_async)

    called = {}

    async def fake_place_order_async(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order_async", fake_place_order_async)
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "update", lambda *a, **k: None)

    asyncio.run(main_module._run_iteration(mem, pf, dry_run=True, take_profit=0.1))

    assert called["args"][0] == "tok"
    assert called["args"][1] == "sell"


@pytest.mark.parametrize(
    "method, target",
    [
        ("onchain", "solhunter_zero.scanner.scan_tokens_onchain"),
        ("websocket", "solhunter_zero.async_scanner.scan_tokens_async"),
        ("pools", "solhunter_zero.scanner.scan_tokens_from_pools"),
        ("file", "solhunter_zero.scanner.scan_tokens_from_file"),
    ],
)
def test_discovery_methods(monkeypatch, method, target):
    called = {}

    async def fake_async(*_a, **_k):
        called["called"] = True
        return []

    def fake_sync(*_a, **_k):
        called["called"] = True
        return []

    if "async" in target:
        monkeypatch.setattr(target, fake_async)
    else:
        monkeypatch.setattr(target, fake_sync)

    import solhunter_zero.scanner as scanner_mod
    monkeypatch.setattr(scanner_mod, "fetch_trending_tokens", lambda: [])
    async def fake_trend():
        return []
    monkeypatch.setattr(scanner_mod, "fetch_trending_tokens_async", fake_trend)

    monkeypatch.setattr(main_module, "run_simulations", lambda token, count=100: [])
    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims: False)
    monkeypatch.setattr(main_module, "place_order_async", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
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

