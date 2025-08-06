import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_investor_demo_trade_types(tmp_path, monkeypatch, dummy_mem):
    reward = 7.0
    metrics = {"loss": [0.0], "rewards": [reward]}

    def stub_run_rl_demo(report_dir: Path) -> float:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "rl_metrics.json").write_text(json.dumps(metrics))
        return reward

    monkeypatch.setattr(investor_demo, "run_rl_demo", stub_run_rl_demo)
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", lambda w, c: w)

    investor_demo.main(["--full-system", "--reports", str(tmp_path)])

    highlights_path = tmp_path / "highlights.json"
    assert highlights_path.exists()
    highlights = json.loads(highlights_path.read_text())

    expected_keys = {
        "arbitrage_path",
        "arbitrage_profit",
        "route_ffi_path",
        "route_ffi_profit",
        "flash_loan_signature",
        "sniper_tokens",
        "dex_new_pools",
        "jito_swaps",
        "rl_reward",
    }
    assert expected_keys <= highlights.keys()
    assert highlights["rl_reward"] == reward

    assert (tmp_path / "rl_metrics.json").exists()
