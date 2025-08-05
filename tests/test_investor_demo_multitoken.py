import importlib
import json
from pathlib import Path

import pytest

from solhunter_zero import investor_demo

pytestmark = pytest.mark.timeout(30)


def test_investor_demo_multitoken(tmp_path, capsys):
    data_path = Path(__file__).resolve().parent / "data" / "prices_multitoken.json"
    # Ensure loader returns mapping
    loaded = investor_demo.load_prices(data_path)
    assert isinstance(loaded, dict)
    assert {"SOL", "ETH"} <= loaded.keys()

    investor_demo.main(["--data", str(data_path), "--reports", str(tmp_path)])
    out = capsys.readouterr().out
    # Output should include token prefixes for strategies
    assert "SOL buy_hold" in out
    assert "ETH buy_hold" in out

    summary = json.loads((tmp_path / "summary.json").read_text())
    tokens = {row["token"] for row in summary}
    assert {"SOL", "ETH"} <= tokens

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert "top_token" in highlights
    assert "top_strategy" in highlights
    top_token = highlights["top_token"]
    top_strategy = highlights["top_strategy"]
    assert any(
        row["token"] == top_token and row["config"] == top_strategy
        for row in summary
    )

    trade_hist = json.loads((tmp_path / "trade_history.json").read_text())
    hist_tokens = {row["token"] for row in trade_hist}
    assert {"SOL", "ETH"} <= hist_tokens
    # Every trade entry should be tagged by token
    assert all("token" in row for row in trade_hist)

    # Plot files should be produced for every token/strategy combination
    if importlib.util.find_spec("matplotlib") is not None:
        for token in ["SOL", "ETH"]:
            for name in ["buy_hold", "momentum", "mean_reversion", "mixed"]:
                assert (
                    tmp_path / f"{token}_{name}.png"
                ).exists(), f"Missing plot {token}_{name}.png"

