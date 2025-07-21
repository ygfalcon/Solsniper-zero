import solhunter_zero.news as news

class FakeResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        pass

SAMPLE_XML = """
<rss><channel>
    <item><title>Good gains ahead</title></item>
    <item><title>Market crash expected</title></item>
</channel></rss>
"""


def test_fetch_headlines(monkeypatch):
    def fake_get(url, timeout=10):
        return FakeResp(SAMPLE_XML)
    monkeypatch.setattr(news.requests, "get", fake_get)
    headlines = news.fetch_headlines(["http://ok"], allowed=["http://ok"])
    assert headlines == ["Good gains ahead", "Market crash expected"]


def test_blocked_feed(monkeypatch):
    called = {}
    def fake_get(url, timeout=10):
        called["url"] = url
        return FakeResp(SAMPLE_XML)
    monkeypatch.setattr(news.requests, "get", fake_get)
    headlines = news.fetch_headlines(["http://bad"], allowed=["http://ok"])
    assert headlines == []
    assert "url" not in called


def test_compute_sentiment():
    text = "good gain up"
    score = news.compute_sentiment(text)
    assert score > 0


def test_fetch_sentiment(monkeypatch):
    def fake_get(url, timeout=10):
        return FakeResp(SAMPLE_XML)
    monkeypatch.setattr(news.requests, "get", fake_get)
    score = news.fetch_sentiment(["http://ok"], allowed=["http://ok"])
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
