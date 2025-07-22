import asyncio

from solhunter_zero.agents.discovery import DiscoveryAgent


async def fake_stream(url, **_):
    yield {"address": "tok", "score": 12.0}


def test_stream_mempool_events(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.agents.discovery.stream_ranked_mempool_tokens",
        fake_stream,
    )

    agent = DiscoveryAgent()

    async def run():
        gen = agent.stream_mempool_events("ws://node")
        data = await asyncio.wait_for(anext(gen), timeout=0.1)
        await gen.aclose()
        return data

    data = asyncio.run(run())
    assert data["address"] == "tok"
