import subprocess
import sys
from solhunter_zero.memory import Memory
from solhunter_zero.trade_analyzer import TradeAnalyzer


def run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "solhunter_zero.backtest_cli", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _setup_memory(path):
    mem = Memory(f"sqlite:///{path}")
    mem.log_trade(token="tok", direction="buy", amount=1, price=1, reason="a1")
    mem.log_trade(token="tok", direction="sell", amount=1, price=2, reason="a1")
    mem.log_trade(token="tok", direction="buy", amount=1, price=2, reason="a2")
    mem.log_trade(token="tok", direction="sell", amount=1, price=1, reason="a2")
    return mem


def test_trade_analyzer_recommend_weights(tmp_path):
    db = tmp_path / "m.db"
    mem = _setup_memory(db)
    analyzer = TradeAnalyzer(mem)
    res = analyzer.recommend_weights({"a1": 1.0, "a2": 1.0})
    assert res["a1"] > 1.0
    assert res["a2"] < 1.0


def test_cli_analyze_trades(tmp_path):
    db = tmp_path / "m.db"
    _setup_memory(db)

    cfg = tmp_path / "cfg.toml"
    cfg.write_text("""[agent_weights]\na1 = 1.0\na2 = 1.0\n""")

    out = tmp_path / "out.toml"
    run_cli([
        "--analyze-trades",
        "--memory",
        f"sqlite:///{db}",
        "-c",
        str(cfg),
        "--weights-out",
        str(out),
    ])

    data = {line.split("=")[0].strip(): float(line.split("=")[1]) for line in out.read_text().splitlines()}
    assert data["a1"] > 1.0
    assert data["a2"] < 1.0
