import json
import subprocess
import sys
from solhunter_zero.backtester import backtest_configs, DEFAULT_STRATEGIES


def run_cli(args):
    return subprocess.run([
        sys.executable,
        "-m",
        "solhunter_zero.backtest_cli",
        *args,
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_backtest_cli_with_dates_and_config(tmp_path):
    history = [
        {"date": "2023-01-01", "price": 1.0},
        {"date": "2023-01-02", "price": 1.2},
        {"date": "2023-01-03", "price": 1.0},
        {"date": "2023-01-04", "price": 1.5},
    ]
    h_path = tmp_path / "h.json"
    h_path.write_text(json.dumps(history))

    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("""[agent_weights]
buy_hold = 1.0
momentum = 1.0
""")

    result = run_cli([
        str(h_path),
        "-c",
        str(cfg_path),
        "--start",
        "2023-01-02",
        "--end",
        "2023-01-04",
        "--strategy",
        "buy_hold",
        "--strategy",
        "momentum",
    ])

    lines = result.stdout.strip().splitlines()
    assert lines
    prices = [1.2, 1.0, 1.5]
    res = backtest_configs(
        prices,
        [("cfg.toml", {"buy_hold": 1.0, "momentum": 1.0})],
        DEFAULT_STRATEGIES,
    )[0]
    assert f"ROI={res.roi:.4f}" in lines[0]
    assert f"Sharpe={res.sharpe:.2f}" in lines[0]
