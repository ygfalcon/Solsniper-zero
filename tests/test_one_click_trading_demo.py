import json
import math
from pathlib import Path

from solhunter_zero import investor_demo


def test_aggregated_summary_exists(tmp_path: Path) -> None:
    """Ensure aggregated metrics file is created with key statistics."""
    investor_demo.main(["--reports", str(tmp_path)])

    agg_path = tmp_path / "aggregated_summary.json"
    summary_path = tmp_path / "summary.json"
    assert agg_path.exists(), "aggregated_summary.json not found"
    assert summary_path.exists(), "summary.json not found"

    aggregated = json.loads(agg_path.read_text())
    summary = json.loads(summary_path.read_text())

    # total_roi and average_sharpe should be finite floats
    for key in ["total_roi", "average_sharpe"]:
        val = aggregated[key]
        assert isinstance(val, float)
        assert math.isfinite(val)

    strategies = {entry["config"] for entry in summary}
    for key in ["best_strategy", "worst_strategy"]:
        assert aggregated[key] in strategies

    for entry in summary:
        assert "roi" in entry and "sharpe" in entry
