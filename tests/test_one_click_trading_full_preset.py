import json
from pathlib import Path

from solhunter_zero import investor_demo


def test_one_click_trading_full_preset(tmp_path: Path, monkeypatch) -> None:
    async def fake_arbitrage() -> dict:
        investor_demo.used_trade_types.add("arbitrage")
        return {}

    async def fake_flash_loan() -> None:
        investor_demo.used_trade_types.add("flash_loan")
        return None

    async def fake_sniper() -> list[str]:
        investor_demo.used_trade_types.add("sniper")
        return []

    async def fake_dex_scanner() -> list[str]:
        investor_demo.used_trade_types.add("dex_scanner")
        return []

    async def fake_route() -> dict:
        investor_demo.used_trade_types.add("route_ffi")
        return {}

    async def fake_jito() -> list[str]:
        investor_demo.used_trade_types.add("jito_stream")
        return []

    monkeypatch.setattr(investor_demo, "_demo_arbitrage", fake_arbitrage)
    monkeypatch.setattr(investor_demo, "_demo_flash_loan", fake_flash_loan)
    monkeypatch.setattr(investor_demo, "_demo_sniper", fake_sniper)
    monkeypatch.setattr(investor_demo, "_demo_dex_scanner", fake_dex_scanner)
    monkeypatch.setattr(investor_demo, "_demo_route_ffi", fake_route)
    monkeypatch.setattr(investor_demo, "_demo_jito_stream", fake_jito)
    monkeypatch.setattr(investor_demo, "_demo_rl_agent", lambda: 0.0)
    investor_demo.used_trade_types.clear()

    monkeypatch.setattr(investor_demo, "RL_REPORT_DIR", tmp_path)

    investor_demo.main(["--preset", "full", "--reports", str(tmp_path)])

    assert (tmp_path / "aggregated_summary.json").exists()
    assert (tmp_path / "aggregate_summary.json").exists()

    summary = json.loads((tmp_path / "summary.json").read_text())
    configs = {row["config"] for row in summary}

    agents_dir = Path(investor_demo.__file__).resolve().parent / "agents"
    strategy_modules = {
        p.stem for p in agents_dir.glob("*.py") if not p.name.startswith("__")
    }

    assert strategy_modules <= configs

