import json

from solhunter_zero import investor_demo


def test_load_prices_fetch(monkeypatch):
    def fake_fetch(token: str = "bitcoin", days: int = 30):
        return [1.0, 2.0, 3.0], ["2020-01-01", "2020-01-02", "2020-01-03"]

    monkeypatch.setattr(investor_demo, "_fetch_live_prices", fake_fetch)
    prices, dates = investor_demo.load_prices(fetch=True)
    assert prices == [1.0, 2.0, 3.0]
    assert dates[0] == "2020-01-01"


def test_load_prices_fetch_fallback(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    data = [{"date": "2020-01-01", "price": 1.0}, {"date": "2020-01-02", "price": 2.0}]
    data_path = tmp_path / "prices.json"
    data_path.write_text(json.dumps(data))
    monkeypatch.setattr(investor_demo, "DATA_FILE", data_path)
    monkeypatch.setattr(investor_demo, "_fetch_live_prices", boom)
    prices, dates = investor_demo.load_prices(fetch=True)
    assert prices == [1.0, 2.0]
    assert dates == ["2020-01-01", "2020-01-02"]
