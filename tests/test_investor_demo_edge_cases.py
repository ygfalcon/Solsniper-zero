import json
import csv

import pytest

from solhunter_zero import investor_demo


pytestmark = pytest.mark.timeout(30)


def test_investor_demo_single_price(tmp_path, monkeypatch, dummy_mem):
    # create minimal dataset with a single price point
    data = [{"date": "2024-01-01", "price": 100.0}]
    data_file = tmp_path / "prices.json"
    data_file.write_text(json.dumps(data))

    # stub Memory to avoid database interactions
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    # run the demo with the minimal dataset
    investor_demo.main([
        "--data",
        str(data_file),
        "--reports",
        str(tmp_path),
        "--capital",
        "100",
    ])

    # ensure trade history files exist and contain unique (strategy, period) pairs
    trade_json_path = tmp_path / "trade_history.json"
    trade_csv_path = tmp_path / "trade_history.csv"
    assert trade_json_path.exists()
    assert trade_csv_path.exists()

    trade_json = json.loads(trade_json_path.read_text())
    with trade_csv_path.open() as cf:
        trade_csv = list(csv.DictReader(cf))

    assert all("date" in e for e in trade_json)
    assert all("date" in r for r in trade_csv)

    pairs_json = {(e["strategy"], e["period"]) for e in trade_json}
    pairs_csv = {(r["strategy"], int(r["period"])) for r in trade_csv}
    assert len(pairs_json) == len(trade_json)
    assert len(pairs_csv) == len(trade_csv)

    # summary metrics should all be zero with final capital unchanged
    summary = json.loads((tmp_path / "summary.json").read_text())
    with (tmp_path / "summary.csv").open() as sf:
        summary_csv = list(csv.DictReader(sf))

    for entry in summary:
        assert entry["roi"] == 0.0
        assert entry["sharpe"] == 0.0
        assert entry["drawdown"] == 0.0
        assert entry["volatility"] == 0.0
        assert entry["trades"] == 0
        assert entry["wins"] == 0
        assert entry["losses"] == 0
        assert entry["final_capital"] == 100.0
    for row in summary_csv:
        assert float(row["roi"]) == 0.0
        assert float(row["sharpe"]) == 0.0
        assert float(row["drawdown"]) == 0.0
        assert float(row["volatility"]) == 0.0
        assert int(row["trades"]) == 0
        assert int(row["wins"]) == 0
        assert int(row["losses"]) == 0
        assert float(row["final_capital"]) == 100.0

    # compute_weighted_returns should yield empty list when all weights are zero
    prices = [100.0, 101.0]
    zero_weights = {"buy_hold": 0.0, "momentum": 0.0, "mean_reversion": 0.0}
    returns = investor_demo.compute_weighted_returns(prices, zero_weights)
    assert returns == []


@pytest.mark.parametrize(
    "data",
    [
        [{"date": "2024-01-01"}],  # missing price
        [{"date": "2024-01-01", "price": "not-a-number"}],  # non-numeric price
    ],
)
def test_load_prices_invalid_data(tmp_path, data):
    data_file = tmp_path / "prices.json"
    data_file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        investor_demo.load_prices(data_file)


def test_investor_demo_empty_dataset(tmp_path):
    data_file = tmp_path / "prices.json"
    data_file.write_text("[]")
    with pytest.raises(ValueError, match="price data must contain at least one entry"):
        investor_demo.main([
            "--data",
            str(data_file),
            "--reports",
            str(tmp_path),
        ])


def test_investor_demo_empty_token_map(tmp_path):
    data_file = tmp_path / "prices.json"
    data_file.write_text("{}")
    with pytest.raises(ValueError, match="price data must contain at least one token"):
        investor_demo.main([
            "--data",
            str(data_file),
            "--reports",
            str(tmp_path),
        ])
