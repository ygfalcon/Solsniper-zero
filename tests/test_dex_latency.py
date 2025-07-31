import importlib
import asyncio
import time
import pytest
import solhunter_zero.arbitrage as arb


def test_refresh_costs_updates_latency(monkeypatch):
    monkeypatch.setenv("MEASURE_DEX_LATENCY", "0")
    mod = importlib.reload(arb)

    latencies = {"jupiter": 0.12}
    monkeypatch.setattr(mod, "measure_dex_latency", lambda urls=None, attempts=3: latencies)
    mod.MEASURE_DEX_LATENCY = True

    _, _, lat = mod.refresh_costs()
    assert lat["jupiter"] == pytest.approx(0.12)
    assert mod.DEX_LATENCY["jupiter"] == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_ping_url_parallel():
    import solhunter_zero.arbitrage as mod

    class DummyResp:
        async def __aenter__(self):
            await asyncio.sleep(0.05)
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def read(self):
            pass

    class DummySession:
        def get(self, *_, **__):
            return DummyResp()

        def ws_connect(self, *_, **__):
            return DummyResp()

    session = DummySession()
    start = time.perf_counter()
    val = await mod._ping_url(session, "http://x", attempts=2)
    duration = time.perf_counter() - start
    assert val == pytest.approx(0.05, rel=0.4)
    assert duration < 0.1


@pytest.mark.asyncio
async def test_measure_latency_parallel(monkeypatch):
    import solhunter_zero.arbitrage as mod

    async def fake_get_session():
        return object()

    calls = []

    async def fake_ping(session, url, attempts=1):
        calls.append(url)
        await asyncio.sleep(0.05)
        return 0.05

    monkeypatch.setattr(mod, "get_session", fake_get_session)
    monkeypatch.setattr(mod, "_ping_url", fake_ping)

    start = time.perf_counter()
    res = await mod.measure_dex_latency_async({"d1": "u1", "d2": "u2"}, attempts=3)
    duration = time.perf_counter() - start
    assert res["d1"] == pytest.approx(0.05)
    assert len(calls) == 6
    assert duration < 0.1
