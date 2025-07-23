import asyncio
from solhunter_zero.agents.mempool_sniper import MempoolSniperAgent


async def fake_stream(url, **_):
    yield {"address": "tok", "liquidity": 5.0, "score": 2.0}


async def fake_prepare(token, side, amount, price, base_url):
    return f"TX_{token}"


async def _run(agent):
    gen = agent.listen("ws://node")
    token = await asyncio.wait_for(anext(gen), timeout=0.1)
    await gen.aclose()
    return token


def test_mempool_sniper_bundles(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.agents.mempool_sniper.stream_ranked_mempool_tokens",
        fake_stream,
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.mempool_sniper._prepare_service_tx", fake_prepare
    )

    sent = []

    async def fake_submit(self, txs):
        sent.append(txs)

    monkeypatch.setattr(
        "solhunter_zero.agents.mempool_sniper.MEVExecutor.submit_bundle",
        fake_submit,
    )

    agent = MempoolSniperAgent(mempool_threshold=1.0, bundle_size=1)
    token = asyncio.run(_run(agent))

    assert token == "tok"
    assert sent == [["TX_tok"]]
