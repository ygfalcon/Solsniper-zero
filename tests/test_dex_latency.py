import importlib
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
