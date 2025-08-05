import json
import re
from pathlib import Path

from solhunter_zero import investor_demo


def test_investor_demo_full_system(tmp_path, monkeypatch, capsys, dummy_mem):
    """Ensure full-system mode uses run_rl_demo and records reward metrics."""

    reward = 5.0
    called = {"value": False}

    def stub_run_rl_demo(report_dir: Path) -> float:
        called["value"] = True
        metrics_file = Path(report_dir) / "rl_metrics.json"
        metrics_file.write_text(json.dumps({"loss": [0.0], "rewards": [reward]}))
        return reward

    monkeypatch.setattr(investor_demo, "run_rl_demo", stub_run_rl_demo)
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", lambda w, c: w)

    investor_demo.main(["--full-system", "--reports", str(tmp_path)])

    assert called["value"], "run_rl_demo was not invoked"

    metrics_path = tmp_path / "rl_metrics.json"
    assert metrics_path.exists()

    out = capsys.readouterr().out
    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade results not printed"
    results = json.loads(match.group(1))
    assert results.get("rl_reward") == reward
    assert f'"rl_reward": {reward}' in out

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == reward
