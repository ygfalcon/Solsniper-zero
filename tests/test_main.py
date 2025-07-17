import pytest
from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult


def test_main_invokes_place_order(monkeypatch):
    # prepare mocks
    monkeypatch.setattr(main_module, "scan_tokens", lambda offline=False: ["tok"])
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(success_prob=1.0, expected_roi=2.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)

    called = {}

    def fake_place_order(token, side, amount, price, testnet=False, dry_run=False, keypair=None):
        called["args"] = (token, side, amount, price, testnet, dry_run, keypair)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order", fake_place_order)

    # avoid DB writes
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "add", lambda *a, **k: None)

    monkeypatch.setattr(main_module.time, "sleep", lambda _: None)

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

    def fake_scan_tokens(*, offline=False):
        recorded["offline"] = offline
        return ["tok"]

    monkeypatch.setattr(main_module, "scan_tokens", fake_scan_tokens)
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(success_prob=1.0, expected_roi=2.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)
    monkeypatch.setattr(main_module, "place_order", lambda *a, **k: {"order_id": "1"})
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "add", lambda *a, **k: None)

    def fake_sleep(_):
        raise SystemExit()

    monkeypatch.setattr(main_module.time, "sleep", fake_sleep)

    with pytest.raises(SystemExit):
        main_module.main(memory_path="sqlite:///:memory:", loop_delay=0, dry_run=True, offline=True)

    assert recorded["offline"] is True

