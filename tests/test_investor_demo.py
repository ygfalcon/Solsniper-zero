import csv
import json
import importlib
import re

import pytest

from solhunter_zero import investor_demo
from solhunter_zero.event_bus import subscribe, publish
from solhunter_zero.schemas import TradeLogged


pytestmark = pytest.mark.timeout(30)


def test_investor_demo(tmp_path, monkeypatch, capsys, dummy_mem):
    monkeypatch.setenv("MEASURE_DEX_LATENCY", "0")
    calls = dummy_mem.calls

    async def log_trade(self, **kwargs):
        self.trade = kwargs
        publish("trade_logged", TradeLogged(**kwargs))

    monkeypatch.setattr(dummy_mem, "log_trade", log_trade)

    def fake_hedge(weights, corrs):
        calls["hedge_called"] = (weights, corrs)
        return weights

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)
    expected_rl = 7.0
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: expected_rl)
    async def fake_arb() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": ["dex1", "dex2"], "profit": 4.795}
    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arb)
    from solhunter_zero import routeffi as rffi
    monkeypatch.setattr(rffi, "_best_route_json", lambda *a, **k: (["r1", "r2"], 1.23))

    events: list[TradeLogged] = []
    unsub = subscribe("trade_logged", lambda p: events.append(p))
    try:
        investor_demo.main(
            [
                "--reports",
                str(tmp_path),
                "--capital",
                "100",
            ]
        )
    finally:
        unsub()

    assert events, "No trade_logged events captured"

    captured = capsys.readouterr()
    out = captured.out
    strategies = {"buy_hold", "momentum", "mean_reversion", "mixed"}
    for name in strategies:
        assert re.search(
            rf"{name}: .*ROI .*Sharpe .*Drawdown .*Win rate", out
        ), f"Missing metrics for {name}"

    match = re.search(r"Trade type results: (\{.*\})", out)
    assert match, "Trade type results not printed"
    trade_results = json.loads(match.group(1))
    assert trade_results["arbitrage_profit"] == pytest.approx(4.795)
    assert trade_results["arbitrage_path"] == ["dex1", "dex2"]
    assert trade_results["route_ffi_path"] == ["r1", "r2"]
    assert trade_results["route_ffi_profit"] == pytest.approx(1.23)
    assert trade_results["flash_loan_signature"] == "demo_sig"
    assert trade_results["sniper_tokens"] == ["TKN"]
    assert trade_results["dex_new_pools"] == ["mintA", "mintB"]
    assert trade_results["rl_reward"] == expected_rl

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
    prices, dates = investor_demo.load_prices(preset="short")
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
            if key in {"config", "token"}:
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
    assert all({"capital", "action", "price", "date"} <= set(r.keys()) for r in trade_rows)
    first = trade_rows[0]
    assert first["action"] == "buy"
    assert float(first["price"]) == prices[0]
    assert first["date"] == dates[0]
    # Should record capital for periods beyond the starting point
    assert any(int(r["period"]) > 0 for r in trade_rows)
    # Expect at least one sell action in the history
    assert any(r["action"] == "sell" for r in trade_rows)
    # Each configured strategy should appear in trade history
    assert any(r["strategy"] == "mean_reversion" for r in trade_rows)

    trade_data = json.loads(trade_json.read_text())
    assert trade_data, "Trade history JSON empty"
    assert all(
        "capital" in r and "action" in r and "price" in r and "date" in r
        for r in trade_data
    )
    first = trade_data[0]
    assert first["action"] == "buy"
    assert first["price"] == prices[0]
    assert first["date"] == dates[0]
    assert any(r["period"] > 0 for r in trade_data)
    assert any(r["action"] == "sell" for r in trade_data)
    assert any(r["strategy"] == "mean_reversion" for r in trade_data)

    highlights_path = tmp_path / "highlights.json"
    corr_path = tmp_path / "correlations.json"
    hedged_path = tmp_path / "hedged_weights.json"
    assert highlights_path.exists(), "Highlights JSON not generated"
    assert corr_path.exists(), "Correlations JSON not generated"
    assert hedged_path.exists(), "Hedged weights JSON not generated"

    corr_data = json.loads(corr_path.read_text())
    hedged_data = json.loads(hedged_path.read_text())
    highlights = json.loads(highlights_path.read_text())

    assert corr_data, "Correlation file empty"
    assert hedged_data, "Hedged weights JSON empty"
    assert highlights, "Highlights JSON empty"

    assert highlights.get("arbitrage_profit") == pytest.approx(4.795)
    assert highlights.get("arbitrage_path") == ["dex1", "dex2"]
    assert highlights.get("route_ffi_path") == ["r1", "r2"]
    assert highlights.get("route_ffi_profit") == pytest.approx(1.23)
    assert highlights.get("flash_loan_signature") == "demo_sig"
    assert highlights.get("sniper_tokens") == ["TKN"]
    assert highlights.get("dex_new_pools") == ["mintA", "mintB"]
    assert highlights.get("rl_reward") == expected_rl

    def _norm(key: str) -> str:
        m = re.match(r"\('([^']+)', '([^']+)'\)", key)
        assert m, f"Unexpected correlation key format: {key}"
        return f"{m.group(1)}-{m.group(2)}"

    corr_pairs = {_norm(k): v for k, v in corr_data.items()}
    top_corr = sorted(corr_pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    corr_summary = {k: v for k, v in top_corr}

    assert highlights.get("key_correlations") == corr_summary
    assert highlights.get("hedged_weights") == hedged_data

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
            assert (
                tmp_path / f"demo_{name}.png"
            ).exists(), f"Missing plot demo_{name}.png"

    # Demo should exercise all trade types including route FFI
    assert {
        "arbitrage",
        "flash_loan",
        "sniper",
        "dex_scanner",
        "route_ffi",
    } <= investor_demo.used_trade_types

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

    async def fake_route() -> None:
        investor_demo.used_trade_types.add("route_ffi")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route)

    investor_demo.main(["--reports", str(tmp_path)])

    assert seen_before == [set()]
    assert investor_demo.used_trade_types == {
        "arbitrage",
        "flash_loan",
        "sniper",
        "dex_scanner",
        "route_ffi",
    }


def test_investor_demo_custom_data_length(tmp_path, monkeypatch):
    price_data = [
        {"date": f"2024-01-0{i}", "price": p}
        for i, p in enumerate([1.0, 2.0, 3.0, 4.0], start=1)
    ]
    data_path = tmp_path / "prices.json"
    data_path.write_text(json.dumps(price_data))

    orig_load = investor_demo.load_prices

    def _load(path, preset=None):
        return orig_load(path=path, preset=None)

    monkeypatch.setattr(investor_demo, "load_prices", _load)

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
