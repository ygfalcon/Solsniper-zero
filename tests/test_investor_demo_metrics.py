import csv
import json
import sys
import types

import pytest
from solhunter_zero import investor_demo
import solhunter_zero.resource_monitor as rm


def _patch_metrics(monkeypatch):
    monkeypatch.setattr(rm, "get_cpu_usage", lambda: 11.0)
    psutil_stub = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(percent=22.0)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)


def test_resource_metrics_in_highlights(tmp_path, monkeypatch):
    _patch_metrics(monkeypatch)

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert highlights.get("cpu_usage") == 11.0
    assert highlights.get("memory_percent") == 22.0


def test_resource_metrics_stdout(tmp_path, monkeypatch, capsys):
    _patch_metrics(monkeypatch)

    investor_demo.main(["--reports", str(tmp_path)])

    captured = capsys.readouterr()
    assert "CPU: 11.00% Memory: 22.00%" in captured.out


def test_no_resource_metrics_when_psutil_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "psutil", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError):
        __import__("psutil")

    investor_demo.main(["--reports", str(tmp_path)])

    highlights = json.loads((tmp_path / "highlights.json").read_text())
    assert "cpu_usage" not in highlights
    assert "memory_percent" not in highlights

    captured = capsys.readouterr()
    assert "CPU:" not in captured.out
    assert "Memory:" not in captured.out


def test_aggregate_summary(tmp_path, monkeypatch):
    _patch_metrics(monkeypatch)

    investor_demo.main(["--reports", str(tmp_path)])

    agg = json.loads((tmp_path / "aggregate_summary.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    per_token = {}
    for row in summary:
        tok = row["token"]
        if tok not in per_token or row["final_capital"] > per_token[tok]["final_capital"]:
            per_token[tok] = row

    expected = [
        {
            "token": t,
            "strategy": r["config"],
            "roi": r["roi"],
            "sharpe": r["sharpe"],
            "final_capital": r["final_capital"],
        }
        for t, r in per_token.items()
    ]
    exp_roi = sum(r["roi"] for r in expected) / len(expected)
    exp_sharpe = sum(r["sharpe"] for r in expected) / len(expected)
    top_row = max(expected, key=lambda r: r["final_capital"])

    assert agg["per_token"] == expected
    assert agg["global_roi"] == pytest.approx(exp_roi)
    assert agg["global_sharpe"] == pytest.approx(exp_sharpe)
    assert agg["top_token"] == top_row["token"]
    assert agg["top_strategy"] == top_row["strategy"]
    assert agg["top_final_capital"] == pytest.approx(top_row["final_capital"])
    assert agg["top_roi"] == pytest.approx(top_row["roi"])
    assert agg["top_sharpe"] == pytest.approx(top_row["sharpe"])

    with (tmp_path / "aggregate_summary.csv").open() as f:
        csv_rows = list(csv.DictReader(f))

    assert len(csv_rows) == len(expected)

    csv_map = {row["token"]: row for row in csv_rows}
    exp_map = {r["token"]: r for r in expected}
    assert set(csv_map) == set(exp_map)

    for token, exp in exp_map.items():
        row = csv_map[token]
        assert row["strategy"] == exp["strategy"]
        assert float(row["roi"]) == pytest.approx(exp["roi"])
        assert float(row["sharpe"]) == pytest.approx(exp["sharpe"])
        assert float(row["final_capital"]) == pytest.approx(exp["final_capital"])
