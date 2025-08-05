import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import csv
import json

import pytest

from solhunter_zero import investor_demo


def _run_and_check(
    cmd: list[str],
    reports_dir: Path,
    repo_root: Path,
    env: dict[str, str],
    data_path: Path,
) -> None:
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Wrote reports" in result.stdout
    assert "Capital Summary" in result.stdout

    strategies = {"buy_hold", "momentum", "mean_reversion", "mixed"}
    pattern = re.compile(
        r"^(?:(?P<token>\w+)\s+)?"
        r"(?P<strategy>buy_hold|momentum|mean_reversion|mixed):\s*"
        r"(?P<capital>-?\d+(?:\.\d+)?)\s+"
        r"ROI\s+(?P<roi>-?\d+(?:\.\d+)?)\s+"
        r"Sharpe\s+(?P<sharpe>-?\d+(?:\.\d+)?)\s+"
        r"Drawdown\s+(?P<drawdown>-?\d+(?:\.\d+)?)\s+"
        r"Win rate\s+(?P<win_rate>-?\d+(?:\.\d+)?)$",
        re.MULTILINE,
    )
    metric_groups = {
        "final_capital": "capital",
        "roi": "roi",
        "sharpe": "sharpe",
        "drawdown": "drawdown",
        "win_rate": "win_rate",
    }
    matches: dict[tuple[str | None, str], dict[str, float]] = {}
    for m in pattern.finditer(result.stdout):
        matches[(m.group("token"), m.group("strategy"))] = {
            metric: float(m.group(group))
            for metric, group in metric_groups.items()
        }

    loaded_prices = investor_demo.load_prices(data_path)
    if isinstance(loaded_prices, dict):
        tokens = set(loaded_prices.keys())
        for token in tokens:
            for strat in strategies:
                assert (token, strat) in matches, f"Missing {strat} for {token}"
    else:
        assert strategies <= {s for (_, s) in matches.keys()}

    match = re.search(r"Trade type results: (\{.*\})", result.stdout)
    assert match, "Trade type results not reported"
    trade_results = json.loads(match.group(1))
    assert trade_results["arbitrage_profit"] == pytest.approx(0.51)
    assert trade_results["flash_loan_profit"] == pytest.approx(0.001482213438735234)
    assert trade_results["sniper_tokens"] == ["token_2023-01-01"]
    assert trade_results["dex_new_pools"] == ["pool_2023-01-02"]

    summary_json = reports_dir / "summary.json"
    summary_csv = reports_dir / "summary.csv"
    assert summary_json.is_file()
    assert summary_csv.is_file()
    assert summary_csv.stat().st_size > 0
    summary_rows = list(csv.DictReader(summary_csv.open()))
    assert any(r["config"] == "mean_reversion" for r in summary_rows)
    for row in summary_rows:
        assert "win_rate" in row
        float(row["win_rate"])

    summary_data = json.loads(summary_json.read_text())
    if isinstance(loaded_prices, dict):
        summary_map = {
            (row["token"], row["config"]): row for row in summary_data
        }
        for key, printed in matches.items():
            if key in summary_map:
                exp = summary_map[key]
                assert printed["final_capital"] == pytest.approx(
                    exp["final_capital"], abs=0.01
                )
                assert printed["roi"] == pytest.approx(exp["roi"], abs=1e-4)
                assert printed["sharpe"] == pytest.approx(exp["sharpe"], abs=1e-4)
                assert printed["drawdown"] == pytest.approx(
                    exp["drawdown"], abs=1e-4
                )
                assert printed["win_rate"] == pytest.approx(
                    exp["win_rate"], abs=1e-4
                )
    else:
        summary_map = {row["config"]: row for row in summary_data}
        assert strategies <= summary_map.keys()
        for (_, name), printed in matches.items():
            if name in summary_map:
                exp = summary_map[name]
                assert printed["final_capital"] == pytest.approx(
                    exp["final_capital"], abs=0.01
                )
                assert printed["roi"] == pytest.approx(exp["roi"], abs=1e-4)
                assert printed["sharpe"] == pytest.approx(
                    exp["sharpe"], abs=1e-4
                )
                assert printed["drawdown"] == pytest.approx(
                    exp["drawdown"], abs=1e-4
                )
                assert printed["win_rate"] == pytest.approx(
                    exp["win_rate"], abs=1e-4
                )

    trade_history_csv = reports_dir / "trade_history.csv"
    highlights_json = reports_dir / "highlights.json"
    assert trade_history_csv.is_file()
    assert trade_history_csv.stat().st_size > 0
    rows = list(csv.DictReader(trade_history_csv.open()))
    assert rows, "Trade history CSV empty"
    first = rows[0]
    assert first["action"] == "buy"
    if isinstance(loaded_prices, dict):
        tokens = set(loaded_prices.keys())
        assert first["token"] in tokens
        prices0, dates0 = loaded_prices[first["token"]]
        assert float(first["price"]) == prices0[0]
        assert first["date"] == dates0[0]
    else:
        prices0, dates0 = loaded_prices
        assert float(first["price"]) == prices0[0]
        assert first["date"] == dates0[0]
    assert any(r["strategy"] == "mean_reversion" for r in rows)
    assert highlights_json.is_file()
    assert highlights_json.stat().st_size > 0
    highlights_data = json.loads(highlights_json.read_text())
    assert highlights_data.get("arbitrage_profit") == pytest.approx(0.51)
    assert highlights_data.get("flash_loan_profit") == pytest.approx(0.001482213438735234)
    assert highlights_data.get("sniper_tokens") == ["token_2023-01-01"]
    assert highlights_data.get("dex_new_pools") == ["pool_2023-01-02"]
    if isinstance(loaded_prices, dict):
        assert highlights_data.get("top_token") in tokens
        assert "SOL buy_hold" in result.stdout
        assert "ETH buy_hold" in result.stdout

    # Correlation and hedged weight outputs should exist
    corr_json = reports_dir / "correlations.json"
    hedged_json = reports_dir / "hedged_weights.json"
    assert corr_json.is_file()
    assert hedged_json.is_file()
    corr_data = json.loads(corr_json.read_text())
    hedged_data = json.loads(hedged_json.read_text())
    assert "('buy_hold', 'momentum')" in corr_data
    assert {"buy_hold", "momentum"} <= hedged_data.keys()

    shutil.rmtree(reports_dir, ignore_errors=True)


@pytest.mark.timeout(30)
@pytest.mark.integration
@pytest.mark.parametrize(
    "base_cmd",
    [
        [sys.executable, "scripts/investor_demo.py"],
        ["solhunter-demo"],
    ],
)
@pytest.mark.parametrize(
    "data_file",
    ["prices_short.json", "prices_multitoken.json"],
)
def test_investor_demo_cli(base_cmd, data_file, tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    if base_cmd == ["solhunter-demo"]:
        wrapper = tmp_path / "solhunter-demo"
        wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "from scripts.investor_demo import main\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        wrapper.chmod(0o755)
        env = {**env, "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}"}

    out = tmp_path / "run"
    data_path = repo_root / "tests" / "data" / data_file
    cmd = base_cmd + [
        "--data",
        str(data_path),
        "--reports",
        str(out),
    ]
    _run_and_check(cmd, out, repo_root, env, data_path)
