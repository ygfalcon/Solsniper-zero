import csv
import json
import importlib

import pytest

from solhunter_zero import investor_demo


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path, monkeypatch):
    calls: dict[str, object] = {}

    class DummyMem:
        def __init__(self, *a, **k):
            calls["mem_init"] = True

        def log_var(self, value: float):
            calls["mem_log_var"] = value

        async def close(self) -> None:  # pragma: no cover - simple stub
            calls["mem_closed"] = True

    def fake_hedge(weights, corrs):
        calls["hedge_called"] = (weights, corrs)
        return weights

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    investor_demo.main(
        [
            "--reports",
            str(tmp_path),
            "--capital",
            "100",
        ]
    )

    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"

    assert summary_json.exists(), "Summary JSON not generated"
    assert summary_csv.exists(), "Summary CSV not generated"

    content = json.loads(summary_json.read_text())
    csv_rows = list(csv.DictReader(summary_csv.open()))

    # JSON and CSV summaries should contain the same number of entries
    assert len(content) == len(csv_rows)

    # Expect results for all configured strategies
    configs = {entry.get("config") for entry in content}
    assert {"buy_hold", "momentum", "mean_reversion", "mixed"} <= configs

    # Each summary entry must include core metrics
    for entry in content:
        for key in ["roi", "sharpe", "drawdown", "final_capital"]:
            assert key in entry

    # Compute expected metrics for the built-in price dataset
    prices = investor_demo.load_prices()
    start_capital = 100.0
    strat_configs = {
        "buy_hold": {"buy_hold": 1.0},
        "momentum": {"momentum": 1.0},
        "mean_reversion": {"mean_reversion": 1.0},
        "mixed": {
            "buy_hold": 1 / 3,
            "momentum": 1 / 3,
            "mean_reversion": 1 / 3,
        },
    }
    expected: dict[str, dict[str, float]] = {}
    for name, weights in strat_configs.items():
        returns = investor_demo.compute_weighted_returns(prices, weights)
        if returns:
            total = 1.0
            cum: list[float] = []
            for r in returns:
                total *= 1 + r
                cum.append(total)
            roi = cum[-1] - 1
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            vol = variance ** 0.5
            sharpe = mean / vol if vol else 0.0
        else:
            roi = 0.0
            sharpe = 0.0
        dd = investor_demo.max_drawdown(returns)
        final_capital = start_capital * (1 + roi)
        expected[name] = {
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
            "final_capital": final_capital,
        }

    for entry in content:
        exp = expected[entry["config"]]
        assert entry["roi"] == pytest.approx(exp["roi"], rel=1e-6)
        assert entry["sharpe"] == pytest.approx(exp["sharpe"], rel=1e-6)
        assert entry["drawdown"] == pytest.approx(exp["drawdown"], rel=1e-6)
        assert entry["final_capital"] == pytest.approx(
            exp["final_capital"], rel=1e-6
        )

    # Trade history and highlight files should be generated
    trade_csv = tmp_path / "trade_history.csv"
    trade_json = tmp_path / "trade_history.json"
    assert trade_csv.exists() or trade_json.exists(), "Trade history not generated"
    if trade_csv.exists():
        trade_rows = list(csv.DictReader(trade_csv.open()))
        assert trade_rows, "Trade history CSV empty"
        assert all({"capital", "action", "price"} <= set(r.keys()) for r in trade_rows)
        first = trade_rows[0]
        assert first["action"] == "buy"
        assert float(first["price"]) == prices[0]
        # Should record capital for periods beyond the starting point
        assert any(int(r["period"]) > 0 for r in trade_rows)
        # Expect at least one sell action in the history
        assert any(r["action"] == "sell" for r in trade_rows)
        # Each configured strategy should appear in trade history
        assert any(r["strategy"] == "mean_reversion" for r in trade_rows)
    else:
        trade_data = json.loads(trade_json.read_text())
        assert trade_data, "Trade history JSON empty"
        assert all("capital" in r and "action" in r and "price" in r for r in trade_data)
        first = trade_data[0]
        assert first["action"] == "buy"
        assert first["price"] == prices[0]
        assert any(r["period"] > 0 for r in trade_data)
        assert any(r["action"] == "sell" for r in trade_data)
        assert any(r["strategy"] == "mean_reversion" for r in trade_data)

    highlights_path = tmp_path / "highlights.json"
    assert highlights_path.exists(), "Highlights JSON not generated"
    highlights = json.loads(highlights_path.read_text())
    assert highlights, "Highlights JSON empty"

    # Highlight metrics should identify the top performing strategy
    top_strategy = highlights.get("top_strategy")
    top_final_capital = highlights.get("top_final_capital")

    # The highlighted strategy/capital pair must exist within the summary data
    assert any(
        entry["config"] == top_strategy
        and entry["final_capital"] == top_final_capital
        for entry in content
    ), "Highlighted top_strategy/top_final_capital mismatch with summary.json"

    # The highlighted final capital must equal the maximum from the summary
    max_final_capital = max(entry["final_capital"] for entry in content)
    assert (
        top_final_capital == max_final_capital
    ), "Highlighted top_final_capital does not match summary.json maximum"

    # Plot files are produced when matplotlib is installed
    if importlib.util.find_spec("matplotlib") is not None:
        for name in ["buy_hold", "momentum", "mean_reversion", "mixed"]:
            assert (tmp_path / f"{name}.png").exists(), f"Missing plot {name}.png"

    # Demo should exercise arbitrage and flash loan trade types
    assert {"arbitrage", "flash_loan"} <= investor_demo.used_trade_types

    # Memory and portfolio helpers should have been invoked
    assert calls.get("mem_init")
    assert calls.get("mem_log_var") == 0.0
    assert calls.get("mem_closed")
    assert "hedge_called" in calls


def test_used_trade_types_reset(tmp_path, monkeypatch):
    investor_demo.used_trade_types.add("legacy")

    seen_before: list[set[str]] = []

    async def fake_arbitrage() -> None:
        seen_before.append(set(investor_demo.used_trade_types))
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)

    investor_demo.main(["--reports", str(tmp_path)])

    assert seen_before == [set()]
    assert investor_demo.used_trade_types == {"arbitrage", "flash_loan"}


def test_investor_demo_custom_data_length(tmp_path):
    price_data = [{"price": p} for p in [1.0, 2.0, 3.0, 4.0]]
    data_path = tmp_path / "prices.json"
    data_path.write_text(json.dumps(price_data))

    investor_demo.main(["--data", str(data_path), "--reports", str(tmp_path)])

    trade_csv = tmp_path / "trade_history.csv"
    assert trade_csv.exists()
    rows = list(csv.DictReader(trade_csv.open()))
    assert rows

    periods_by_strategy: dict[str, set[int]] = {}
    for row in rows:
        periods_by_strategy.setdefault(row["strategy"], set()).add(
            int(row["period"])
        )

    expected_last = len(price_data) - 1
    for periods in periods_by_strategy.values():
        assert max(periods) == expected_last
        assert len(periods) == len(price_data)
