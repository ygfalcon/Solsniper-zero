import json
from pathlib import Path
import io

import paper


def test_fetch_live_dataset_parses_prices(monkeypatch):
    """_fetch_live_dataset should return file with numeric prices."""

    sample = {"prices": [[0, 1.0], [1, 2.5]]}

    class Dummy(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    resp = Dummy(json.dumps(sample).encode())
    monkeypatch.setattr(paper, "urlopen", lambda *a, **k: resp)

    path = paper._fetch_live_dataset()
    assert path is not None
    data = json.loads(Path(path).read_text())
    assert data and all(isinstance(d["price"], (int, float)) for d in data)


def test_fetch_live_failure_uses_samples(monkeypatch, tmp_path):
    """paper.run should fall back to sample ticks and log trades."""

    def raise_urlopen(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(paper, "urlopen", raise_urlopen)

    called = {"samples": False}

    def fake_load_sample_ticks(path=None):
        called["samples"] = True
        return [{"price": 1.0, "timestamp": 1}, {"price": 2.0, "timestamp": 2}]

    monkeypatch.setattr(paper, "load_sample_ticks", fake_load_sample_ticks)

    def fake_run_simple_bot(dataset, reports, **kwargs):
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "trade_history.json").write_text(
            json.dumps([{ "token": "SOL", "side": "buy", "amount": 1.0, "price": 1.0 }])
        )
        (reports / "summary.json").write_text("{}")
        (reports / "highlights.json").write_text("{}")

    monkeypatch.setattr(paper, "run_simple_bot", fake_run_simple_bot)

    monkeypatch.chdir(tmp_path)
    paper.run(["--fetch-live"])

    assert called["samples"]
    trade_path = Path("reports") / "trade_history.json"
    assert trade_path.exists()
    data = json.loads(trade_path.read_text())
    assert data and isinstance(data, list)
