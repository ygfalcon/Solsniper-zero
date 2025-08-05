import inspect
import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_investor_demo_short_weights(tmp_path, monkeypatch, dummy_mem):
    monkeypatch.setattr(
        investor_demo,
        "configs",
        {"short_mix": {"buy_hold": 1.0, "mean_reversion": -0.5}},
        raising=False,
    )

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    orig_mkdir = Path.mkdir

    def patched_mkdir(self, *args, **kwargs):
        frame = inspect.currentframe().f_back
        if (
            frame
            and frame.f_code.co_name == "main"
            and frame.f_globals.get("__name__") == investor_demo.__name__
        ):
            cfg = getattr(investor_demo, "configs", {})
            if "configs" in frame.f_locals:
                frame.f_locals["configs"].update(cfg)
        return orig_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", patched_mkdir)

    investor_demo.main(
        [
            "--data",
            str(Path("tests/data/prices_short.json")),
            "--reports",
            str(tmp_path),
        ]
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    summary_map = {row["config"]: row for row in summary}
    short_metrics = summary_map["short_mix"]
    assert short_metrics["wins"] > 0 and short_metrics["losses"] > 0

    history = json.loads((tmp_path / "trade_history.json").read_text())
    actions = {
        row["action"]
        for row in history
        if row.get("strategy") == "short_mix"
    }
    assert {"buy", "sell"} <= actions
