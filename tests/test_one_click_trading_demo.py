import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_aggregated_summary_exists(tmp_path: Path) -> None:
    """Ensure aggregated metrics file is created with key statistics."""
    investor_demo.main(["--reports", str(tmp_path)])

    agg_path = tmp_path / "aggregated_summary.json"
    assert agg_path.exists(), "aggregated_summary.json not found"

    data = json.loads(agg_path.read_text())
    for key in ["total_roi", "average_sharpe", "best_strategy", "worst_strategy"]:
        assert key in data
