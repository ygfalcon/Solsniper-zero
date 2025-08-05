import json
import sys
import types

import pytest

from solhunter_zero import investor_demo
import solhunter_zero.resource_monitor as rm


@pytest.mark.timeout(30)
@pytest.mark.parametrize("capital", [100.0, 200.0])
def test_investor_demo_metrics(tmp_path, monkeypatch, dummy_mem, capital):
    async def fake_arbitrage():
        investor_demo.used_trade_types.add("arbitrage")
        return {"path": ["dex1", "dex2"], "profit": 0.0}

    async def fake_flash():
        investor_demo.used_trade_types.add("flash_loan")
        return "sig"

    async def fake_sniper():
        investor_demo.used_trade_types.add("sniper")
        return ["TKN"]

    async def fake_dex():
        investor_demo.used_trade_types.add("dex_scanner")
        return ["pool"]

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    prices, _ = investor_demo.load_prices(preset="short")
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
    expected = {}
    for name, weights in strat_configs.items():
        returns = investor_demo.compute_weighted_returns(prices, weights)
        if returns:
            total = 1.0
            for r in returns:
                total *= 1 + r
            roi = total - 1
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            vol = variance ** 0.5
            sharpe = mean / vol if vol else 0.0
            dd = investor_demo.max_drawdown(returns)
        else:
            roi = sharpe = dd = 0.0
        expected[name] = {
            "roi": roi,
            "sharpe": sharpe,
            "drawdown": dd,
        }

    reports = tmp_path / f"run_{int(capital)}"
    investor_demo.main([
        "--reports",
        str(reports),
        "--capital",
        str(capital),
    ])

    summary = json.loads((reports / "summary.json").read_text())
    metrics = {row["config"]: row for row in summary}
    for name, exp in expected.items():
        row = metrics[name]
        assert row["roi"] == pytest.approx(exp["roi"], rel=1e-6)
        assert row["sharpe"] == pytest.approx(exp["sharpe"], rel=1e-6)
        assert row["drawdown"] == pytest.approx(exp["drawdown"], rel=1e-6)


@pytest.mark.timeout(30)
def test_resource_usage_logging(tmp_path, monkeypatch, dummy_mem, capsys):
    async def fake_arbitrage():
        investor_demo.used_trade_types.add("arbitrage")

    async def fake_flash():
        investor_demo.used_trade_types.add("flash_loan")

    async def fake_sniper():
        investor_demo.used_trade_types.add("sniper")

    async def fake_dex():
        investor_demo.used_trade_types.add("dex_scanner")

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)
    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)

    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 11.0)
    psutil_stub = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(percent=22.0)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    investor_demo.main(["--reports", str(tmp_path)])
    captured = capsys.readouterr()
    assert (
        "Resource usage - CPU: 11.00% Memory: 22.00%" in captured.out
    )

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 11.0
    assert highlights.get("memory_percent") == 22.0
    assert (tmp_path / "summary.csv").exists()
