import pytest
from solhunter_zero import investor_demo


def _manual_corr() -> dict[tuple[str, str], float]:
    prices, _ = investor_demo.load_prices()
    strategy_returns = {name: strat(prices) for name, strat in investor_demo.DEFAULT_STRATEGIES}
    keys = list(strategy_returns.keys())
    out: dict[tuple[str, str], float] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = strategy_returns[keys[i]]
            b = strategy_returns[keys[j]]
            n = min(len(a), len(b))
            if n == 0:
                continue
            a = a[:n]
            b = b[:n]
            ma = sum(a) / n
            mb = sum(b) / n
            va = sum((x - ma) ** 2 for x in a) / n
            vb = sum((y - mb) ** 2 for y in b) / n
            if va <= 0 or vb <= 0:
                c = 0.0
            else:
                cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n)) / n
                c = cov / (va ** 0.5 * vb ** 0.5)
            out[(keys[i], keys[j])] = c
    return out


def test_correlation_matrix_fallback(tmp_path, monkeypatch):
    class DummyMem:
        def __init__(self, *a, **k):
            pass
        def log_var(self, v: float) -> None:
            pass
        async def close(self) -> None:  # pragma: no cover - simple stub
            pass

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)

    def boom(_series):
        raise RuntimeError("boom")

    monkeypatch.setattr(investor_demo, "correlation_matrix", boom)

    calls: dict[str, dict] = {}

    def fake_hedge(weights, corr_pairs):
        calls["weights"] = weights
        calls["corr_pairs"] = corr_pairs
        return weights

    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    expected = _manual_corr()

    try:
        investor_demo.main(["--reports", str(tmp_path)])
    except RuntimeError as exc:  # pragma: no cover - should not happen
        pytest.fail(f"main raised RuntimeError: {exc}")

    assert "corr_pairs" in calls, "hedge_allocation not invoked"
    corr_pairs = calls["corr_pairs"]
    assert corr_pairs, "no correlation pairs computed"
    assert set(corr_pairs.keys()) == set(expected.keys())
    for k, v in expected.items():
        assert corr_pairs[k] == pytest.approx(v)
