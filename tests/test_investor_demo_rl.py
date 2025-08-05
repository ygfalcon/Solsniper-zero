import json
import re

from solhunter_zero import investor_demo


def test_investor_demo_rl_metric(tmp_path, monkeypatch, capsys, dummy_mem):
    """Ensure the RL demo metric is recorded in highlights and stdout."""

    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    expected = 7.0
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: expected)

    investor_demo.main(["--reports", str(tmp_path)])

    out = capsys.readouterr().out
    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade results not printed"
    results = json.loads(match.group(1))
    assert results.get("rl_reward") == expected
    assert f'"rl_reward": {expected}' in out

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == expected

