import pytest
from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult
import asyncio


def test_main_invokes_place_order(monkeypatch):
    # prepare mocks
    async def fake_scan_tokens_async(offline=False):
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

    async def fake_scan_tokens_async(*, offline=False):
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

    async def fake_scan_tokens_async(*, offline=False):
        return []

    monkeypatch.setattr(main_module, "scan_tokens_async", fake_scan_tokens_async)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(0.2, -0.1, volume=200.0, liquidity=400.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: False)
    monkeypatch.setattr(main_module, "should_sell", lambda sims: True)

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

