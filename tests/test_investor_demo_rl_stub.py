import json

from solhunter_zero import investor_demo


def test_run_rl_demo_stub(tmp_path, monkeypatch, dummy_mem):
    """Stub run_rl_demo to ensure metrics and highlights use stubbed values."""

    metrics = {"loss": [1.0, 2.0], "rewards": [0.5, 0.6]}
    reward = 123.45

    def stub(report_dir):
        with open(report_dir / "rl_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh)
        return reward

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "run_rl_demo", stub)

    investor_demo.main(["--reports", str(tmp_path), "--rl-demo"])

    metrics_path = tmp_path / "rl_metrics.json"
    assert json.loads(metrics_path.read_text()) == metrics

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("rl_reward") == reward
