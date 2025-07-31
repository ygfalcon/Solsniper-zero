import asyncio
import pytest

from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero import event_bus


def test_execution_agent_cpu_adjust(monkeypatch):
    monkeypatch.setenv("EWMA_ALPHA", "1.0")
    monkeypatch.setenv("STEP_SIZE", "0.5")
    agent = ExecutionAgent(rate_limit=0.1, concurrency=4, min_rate=0.05, max_rate=0.2)
    # high CPU usage should gradually lower concurrency and increase rate limit
    event_bus.publish("system_metrics_combined", {"cpu": 90.0})
    assert agent._sem._value == 2
    assert agent.rate_limit > 0.1
    # second update moves closer to target
    event_bus.publish("system_metrics_combined", {"cpu": 90.0})
    assert agent._sem._value == 1
    assert agent.rate_limit > 0.13
    # low CPU restores limits gradually
    event_bus.publish("system_metrics_combined", {"cpu": 0.0})
    assert agent._sem._value >= 2
    assert agent.rate_limit < 0.13


async def fake_order(*_a, **_k):
    await asyncio.sleep(0.01)
    return {}


def test_execution_agent_cpu_benchmark(monkeypatch):
    monkeypatch.setenv("EWMA_ALPHA", "1.0")
    monkeypatch.setenv("STEP_SIZE", "0.5")
    monkeypatch.setattr(
        "solhunter_zero.agents.execution.place_order_async",
        fake_order,
    )

    agent = ExecutionAgent(rate_limit=0.0, concurrency=4, min_rate=0.0, max_rate=0.0)

    event_bus.publish("system_metrics_combined", {"cpu": 90.0})

    async def run():
        start = asyncio.get_event_loop().time()
        await asyncio.gather(
            *(
                agent.execute({"token": "t", "side": "buy", "amount": 0.0, "price": 0.0})
                for _ in range(4)
            )
        )
        return asyncio.get_event_loop().time() - start

    high = asyncio.run(run())
    event_bus.publish("system_metrics_combined", {"cpu": 0.0})
    low = asyncio.run(run())
    assert high > low
