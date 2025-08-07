import json

import pytest

from solhunter_zero import investor_demo

TOKENS = {"SOL", "ETH", "BTC"}


@pytest.mark.timeout(30)
def test_investor_demo_multi_token_top_fields(tmp_path):
    investor_demo.main(["--reports", str(tmp_path), "--preset", "multi"])

    agg = json.loads((tmp_path / "aggregate_summary.json").read_text())
    highlights = json.loads((tmp_path / "highlights.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert "top_token" in agg
    assert "top_token" in highlights

    per_token = {row["token"] for row in agg.get("per_token", [])}
    summary_tokens = {row["token"] for row in summary}

    assert TOKENS <= per_token
    assert TOKENS <= summary_tokens
