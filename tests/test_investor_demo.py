import csv
import json
import importlib
import re

import pytest

from solhunter_zero import investor_demo


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path, monkeypatch, capsys):
    calls: dict[str, object] = {}

    class DummyMem:
        def __init__(self, *a, **k):
            calls["mem_init"] = True
            self.trade: dict | None = None

        async def log_trade(self, **kwargs):
            self.trade = kwargs

        async def list_trades(self, token: str):
            return [self.trade] if self.trade else []

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

    captured = capsys.readouterr()
    out = captured.out
    strategies = {"buy_hold", "momentum", "mean_reversion", "mixed"}
    for name in strategies:
        assert re.search(
            rf"{name}: .*ROI .*Sharpe .*Drawdown .*Win rate", out
        ), f"Missing metrics for {name}"

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
    required = [
        "roi",
        "sharpe",
        "drawdown",
        "volatility",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "final_capital",
    ]
    for entry in content:
        for key in required:
            assert key in entry
    for row in csv_rows:
        for key in required:
            assert key in row

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
    expected: dict[str, dict[str, float | int]] = {}
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
            trades = sum(1 for r in returns if r != 0)
            wins = sum(1 for r in returns if r > 0)
            losses = sum(1 for r in returns if r < 0)
            win_rate = wins / trades if trades else 0.0
        else:
            roi = 0.0
            sharpe = 0.0
            vol = 0.0
            trades = wins = losses = 0
            win_rate = 0.0
        dd = investor_demo.max_drawdown(returns)
        final_capital = start_capital * (1 + roi)
        expected[name] = {
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
            "volatility": vol,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "final_capital": final_capital,
        }
    csv_map: dict[str, dict[str, float | int | str]] = {}
    for row in csv_rows:
        parsed: dict[str, float | int | str] = {}
        for key, val in row.items():
            if key == "config":
                parsed[key] = val
            elif key in {"trades", "wins", "losses"}:
                parsed[key] = int(val)
            else:
                parsed[key] = float(val)
        csv_map[row["config"]] = parsed

    for entry in content:
        exp = expected[entry["config"]]
        csv_entry = csv_map[entry["config"]]
        for key in ["roi", "sharpe", "drawdown", "volatility", "win_rate", "final_capital"]:
            assert entry[key] == pytest.approx(exp[key], rel=1e-6)
            assert csv_entry[key] == pytest.approx(exp[key], rel=1e-6)
        for key in ["trades", "wins", "losses"]:
            assert entry[key] == exp[key]
            assert csv_entry[key] == exp[key]

    # Trade history and highlight files should be generated
    trade_csv = tmp_path / "trade_history.csv"
    trade_json = tmp_path / "trade_history.json"
    assert trade_csv.exists(), "Trade history CSV not generated"
    assert trade_json.exists(), "Trade history JSON not generated"

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

    # Correlation and hedged weight outputs should be generated
    corr_path = tmp_path / "correlations.json"
    hedged_path = tmp_path / "hedged_weights.json"
    assert corr_path.exists(), "Correlations JSON not generated"
    assert hedged_path.exists(), "Hedged weights JSON not generated"
    corr_data = json.loads(corr_path.read_text())
    hedged_data = json.loads(hedged_path.read_text())
    assert "buy_hold-momentum" in corr_data
    assert {"buy_hold", "momentum"} <= hedged_data.keys()
    assert hedged_data["buy_hold"] + hedged_data["momentum"] == pytest.approx(1.0)

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

    # Demo should exercise arbitrage, flash loan, sniper and DEX scanner trade types
    assert {"arbitrage", "flash_loan", "sniper", "dex_scanner"} <= investor_demo.used_trade_types

    # Memory and portfolio helpers should have been invoked
    assert calls.get("mem_init")
    assert calls.get("mem_log_var") == 0.0
    assert calls.get("mem_closed")
    assert "hedge_called" in calls
    _, corr_map = calls["hedge_called"]
    assert corr_map, "hedge_allocation should receive correlations"


def test_used_trade_types_reset(tmp_path, monkeypatch):
    investor_demo.used_trade_types.add("legacy")

    seen_before: list[set[str]] = []

    async def fake_arbitrage() -> None:
        seen_before.append(set(investor_demo.used_trade_types))
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash() -> None:
        investor_demo.used_trade_types.add("flash_loan")

    async def fake_sniper() -> None:
        investor_demo.used_trade_types.add("sniper")

    async def fake_dex() -> None:
        investor_demo.used_trade_types.add("dex_scanner")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)

    investor_demo.main(["--reports", str(tmp_path)])

    assert seen_before == [set()]
    assert investor_demo.used_trade_types == {"arbitrage", "flash_loan", "sniper", "dex_scanner"}


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
