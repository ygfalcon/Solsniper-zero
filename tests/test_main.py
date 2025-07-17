import pytest
from solhunter_zero import main as main_module
from solhunter_zero.simulation import SimulationResult


def test_main_invokes_place_order(monkeypatch):
    # prepare mocks
    monkeypatch.setattr(main_module, "scan_tokens", lambda: ["tok"])
    monkeypatch.setattr(
        main_module,
        "run_simulations",
        lambda token, count=100: [SimulationResult(success_prob=1.0, expected_roi=2.0)],
    )
    monkeypatch.setattr(main_module, "should_buy", lambda sims: True)

    called = {}

    def fake_place_order(token, side, amount, price, testnet=False, dry_run=False):
        called["args"] = (token, side, amount, price, testnet, dry_run)
        return {"order_id": "1"}

    monkeypatch.setattr(main_module, "place_order", fake_place_order)

    # avoid DB writes
    monkeypatch.setattr(main_module.Memory, "log_trade", lambda *a, **k: None)
    monkeypatch.setattr(main_module.Portfolio, "add", lambda *a, **k: None)

    def fake_sleep(_):
        raise SystemExit()

    monkeypatch.setattr(main_module.time, "sleep", fake_sleep)

    with pytest.raises(SystemExit):
        main_module.main(memory_path="sqlite:///:memory:", loop_delay=0, dry_run=True)

    assert called["args"][-1] is True
    assert called["args"][0] == "tok"
