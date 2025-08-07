import re

from solhunter_zero import investor_demo


def test_learning_loop_outputs(tmp_path, monkeypatch, capsys, dummy_mem):
    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", dummy_mem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    rewards = iter([0.0, 1.0, 2.0, 3.0])

    def fake_rl_agent():
        return next(rewards)

    monkeypatch.setattr(investor_demo, "_demo_rl_agent", fake_rl_agent)

    async def fake_arbitrage():
        investor_demo.used_trade_types.add("arbitrage")
        return {}

    async def fake_flash():
        investor_demo.used_trade_types.add("flash_loan")
        return None

    async def fake_sniper():
        investor_demo.used_trade_types.add("sniper")
        return None

    async def fake_dex():
        investor_demo.used_trade_types.add("dex_scanner")
        return None

    async def fake_route():
        investor_demo.used_trade_types.add("route_ffi")
        return {}

    async def fake_jito():
        investor_demo.used_trade_types.add("jito_stream")
        return []

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route)
    monkeypatch.setattr(investor_demo, "_demo_jito_stream", fake_jito)

    investor_demo.main(["--reports", str(tmp_path), "--learn"])

    out = capsys.readouterr().out
    pattern = r"Learning iteration \d+: reward ([0-9.]+) portfolio [0-9.]+ weights (\{[^}]+\})"
    lines = re.findall(pattern, out)
    assert len(lines) == 3
    assert float(lines[0][0]) == 1.0 and "buy_hold" in lines[0][1]
    assert float(lines[1][0]) == 2.0 and "momentum" in lines[1][1]
    assert float(lines[2][0]) == 3.0 and "mean_reversion" in lines[2][1]
