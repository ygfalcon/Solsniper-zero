from pathlib import Path
from solhunter_zero.backtester import backtest_configs, DEFAULT_STRATEGIES
from solhunter_zero import backtest_cli


def test_backtest_cli_with_dates_and_config(tmp_path):
    h_path = Path(__file__).parent / "data" / "prices.json"
    prices = backtest_cli._load_history(str(h_path), "2023-01-02", "2023-01-04")
    assert prices == [101.2, 101.05, 100.56]

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("""[agent_weights]
buy_hold = 1.0
momentum = 1.0
""")

    res = backtest_configs(
        prices,
        [("cfg.toml", {"buy_hold": 1.0, "momentum": 1.0})],
        DEFAULT_STRATEGIES,
    )[0]
    assert res.roi != 0.0
    assert res.sharpe == res.sharpe
