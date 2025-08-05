import json
import re
import types
import sys

from solhunter_zero import investor_demo


def test_investor_demo_rl_metric(tmp_path, monkeypatch, capsys):
    """Ensure the RL demo metric is recorded in highlights and stdout."""

    class DummyMem:
        def __init__(self, *a, **k):
            self.trade = None

        async def log_trade(self, **kwargs):
            self.trade = kwargs

        async def list_trades(self, token: str):  # pragma: no cover - simple stub
            return [self.trade] if self.trade else []

        def log_var(self, value: float) -> None:
            pass

        async def close(self) -> None:  # pragma: no cover - simple stub
            pass

    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    expected = 7.0
    rl_stub = types.SimpleNamespace(train_step=lambda *a, **k: expected)
    monkeypatch.setitem(sys.modules, "solhunter_zero.ppo_agent", rl_stub)

    investor_demo.main(["--reports", str(tmp_path)])

    out = capsys.readouterr().out
    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade results not printed"
    results = json.loads(match.group(1))
    assert results.get("rl_reward") == expected

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == expected

