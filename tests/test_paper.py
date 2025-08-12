"""Tests for the paper trading CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import paper
import solhunter_zero.simple_bot as simple_bot


def test_paper_fetches_url_and_forwards(tmp_path, monkeypatch):
    """``paper.run`` should download data and forward arguments."""

    dataset = json.dumps([
        {"date": "2024-01-01", "price": 1.0},
        {"date": "2024-01-02", "price": 2.0},
    ])

    class DummyResp:
        def __enter__(self):  # pragma: no cover - trivial
            return self

        def __exit__(self, *exc):  # pragma: no cover - trivial
            return False

        def read(self) -> bytes:
            return dataset.encode("utf-8")

    monkeypatch.setattr(simple_bot, "urlopen", lambda *a, **k: DummyResp())

    called: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        called["args"] = list(args)

    monkeypatch.setattr(simple_bot.investor_demo, "main", fake_main)

    reports = tmp_path / "reports"
    paper.run(["--reports", str(reports), "--url", "http://example"])

    forwarded = called["args"]
    assert forwarded[:2] == ["--reports", str(reports)]
    assert "--data" in forwarded
    data_path = Path(forwarded[forwarded.index("--data") + 1])
    assert json.loads(data_path.read_text()) == json.loads(dataset)


def test_paper_forwards_preset(tmp_path, monkeypatch):
    """Providing ``--preset`` should be forwarded to the demo."""

    called: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        called["args"] = list(args)

    monkeypatch.setattr(simple_bot.investor_demo, "main", fake_main)

    reports = tmp_path / "out"
    paper.run(["--reports", str(reports), "--preset", "short"])

    assert called["args"] == ["--reports", str(reports), "--preset", "short"]

