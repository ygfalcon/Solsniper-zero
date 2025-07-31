import time
from collections import deque

import solhunter_zero.metrics_aggregator as ma


def test_metrics_aggregator_average(monkeypatch):
    outputs = []
    monkeypatch.setattr(ma, "publish", lambda t, p: outputs.append((t, p)))
    monkeypatch.setattr(ma, "_HISTORY", deque(maxlen=4))
    monkeypatch.setattr(ma, "_CPU_TOTAL", 0.0)
    monkeypatch.setattr(ma, "_MEM_TOTAL", 0.0)

    events = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0), (5.0, 50.0)]
    expected = []
    hist = deque(maxlen=4)
    for cpu, mem in events:
        hist.append((cpu, mem))
        avg_cpu = sum(c for c, _ in hist) / len(hist)
        avg_mem = sum(m for _, m in hist) / len(hist)
        expected.append({"cpu": avg_cpu, "memory": avg_mem})
        ma._on_metrics({"cpu": cpu, "memory": mem})

    assert [p[1] for p in outputs] == expected


def _measure_old(n: int) -> float:
    hist = deque(maxlen=4)
    start = time.perf_counter()
    for i in range(n):
        cpu = float(i % 100)
        mem = float(i % 50)
        hist.append((cpu, mem))
        _ = sum(c for c, _ in hist) / len(hist)
        _ = sum(m for _, m in hist) / len(hist)
    return time.perf_counter() - start


def _measure_new(monkeypatch, n: int) -> float:
    monkeypatch.setattr(ma, "_HISTORY", deque(maxlen=4))
    monkeypatch.setattr(ma, "_CPU_TOTAL", 0.0)
    monkeypatch.setattr(ma, "_MEM_TOTAL", 0.0)
    monkeypatch.setattr(ma, "publish", lambda *_a, **_k: None)

    start = time.perf_counter()
    for i in range(n):
        ma._on_metrics({"cpu": float(i % 100), "memory": float(i % 50)})
    return time.perf_counter() - start


def test_metrics_aggregator_benchmark(monkeypatch):
    n = 10000
    old_t = _measure_old(n)
    new_t = _measure_new(monkeypatch, n)
    assert new_t <= old_t
