import csv
import json

import pytest
from solhunter_zero import investor_demo


@pytest.mark.timeout(30)
def test_investor_demo_aggregate_reports(tmp_path):
    """Ensure aggregate summary files are generated and consistent."""
    investor_demo.main(["--reports", str(tmp_path)])

    json_path = tmp_path / "aggregate_summary.json"
    csv_path = tmp_path / "aggregate_summary.csv"

    assert json_path.exists(), "aggregate_summary.json not found"
    assert csv_path.exists(), "aggregate_summary.csv not found"

    agg = json.loads(json_path.read_text())
    assert "global_roi" in agg
    assert "global_sharpe" in agg

    per_token = agg.get("per_token", [])
    with csv_path.open() as f:
        csv_rows = list(csv.DictReader(f))

    assert len(csv_rows) == len(per_token)

    csv_map = {row["token"]: row for row in csv_rows}
    token_map = {row["token"]: row for row in per_token}
    assert set(csv_map) == set(token_map)

    for token, expected in token_map.items():
        row = csv_map[token]
        assert row["strategy"] == expected["strategy"]
        assert float(row["roi"]) == pytest.approx(expected["roi"])
        assert float(row["sharpe"]) == pytest.approx(expected["sharpe"])
        assert float(row["final_capital"]) == pytest.approx(expected["final_capital"])
