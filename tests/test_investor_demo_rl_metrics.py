import json
import re

from solhunter_zero import investor_demo


def test_rl_metrics_written(tmp_path, monkeypatch, capsys, dummy_mem):
    """Ensure RL demo writes metrics and reports reward."""

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    investor_demo.main(["--rl-demo", "--reports", str(tmp_path)])

    metrics_file = tmp_path / "rl_metrics.json"
    assert metrics_file.exists()

    metrics = json.loads(metrics_file.read_text())
    assert metrics.get("loss") and metrics.get("rewards")
    assert len(metrics["loss"]) > 0
    assert len(metrics["rewards"]) > 0

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    reward = highlights.get("rl_reward")
    assert reward is not None and reward != 0

    out = capsys.readouterr().out
    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade results not printed"
    results = json.loads(match.group(1))
    assert results.get("rl_reward") == reward
    assert reward != 0
    assert f'"rl_reward": {reward}' in out
