import asyncio
from solhunter_zero.agents.mev_sandwich import MEVSandwichAgent


async def fake_stream(url, **_):
    yield {"address": "tok", "avg_swap_size": 2.0}


async def _run(agent):
    gen = agent.listen("ws://node")
    token = await asyncio.wait_for(anext(gen), timeout=0.1)
    await gen.aclose()
    return token


def test_mev_sandwich_bundle(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich.stream_ranked_mempool_tokens_with_depth",
        fake_stream,
    )

    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich.fetch_slippage_onchain",
        lambda t, u: 0.3,
    )

    async def fake_fetch(token, side, amount, price, base_url):
        return f"MSG_{side}"

    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich._fetch_swap_tx_message",
        fake_fetch,
    )

    async def fake_prepare(msg):
        return f"TX_{msg}"

    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich.prepare_signed_tx",
        fake_prepare,
    )

    sent = []

    async def fake_submit(self, txs):
        sent.append(txs)

    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich.MEVExecutor.submit_bundle",
        fake_submit,
    )

    agent = MEVSandwichAgent(size_threshold=1.0, slippage_threshold=0.2)
    token = asyncio.run(_run(agent))

    assert token == "tok"
    assert sent == [["TX_MSG_buy", "TX_MSG_sell"]]
