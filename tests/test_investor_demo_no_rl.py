import json

from solhunter_zero import investor_demo


def test_no_rl_metrics(tmp_path, monkeypatch, dummy_mem):
    """Ensure no RL metrics are produced when RL demo is disabled."""

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    investor_demo.main(["--reports", str(tmp_path)])

    metrics_file = tmp_path / "rl_metrics.json"
    assert not metrics_file.exists()

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == 0.0
