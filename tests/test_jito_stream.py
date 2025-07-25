import asyncio

from solhunter_zero.agents.mev_sandwich import MEVSandwichAgent
from solhunter_zero.agents.flashloan_sandwich import FlashloanSandwichAgent


async def fake_jito(url, *, auth=None):
    yield {"token": "tok", "size": 2.0, "slippage": 0.3}


async def _run(agent):
    gen = agent.listen("ws://node")
    token = await asyncio.wait_for(anext(gen), timeout=0.1)
    await gen.aclose()
    return token


def test_mev_sandwich_from_jito(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.jito_stream.stream_pending_transactions", fake_jito
    )

    async def fake_fetch(token, side, amount, price, base_url):
        return f"MSG_{side}"

    monkeypatch.setattr(
        "solhunter_zero.agents.mev_sandwich._fetch_swap_tx_message", fake_fetch
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

    agent = MEVSandwichAgent(jito_rpc_url="ws://jito", jito_auth="A")
    token = asyncio.run(_run(agent))

    assert token == "tok"
    assert sent == [["TX_MSG_buy", "TX_MSG_sell"]]


def test_flashloan_sandwich_from_jito(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.jito_stream.stream_pending_transactions", fake_jito
    )

    async def fake_fetch(token, side, amount, price, base_url):
        return f"MSG_{side}"

    monkeypatch.setattr(
        "solhunter_zero.agents.flashloan_sandwich._fetch_swap_tx_message",
        fake_fetch,
    )
    async def fake_prepare2(msg):
        return f"TX_{msg}"

    monkeypatch.setattr(
        "solhunter_zero.agents.flashloan_sandwich.prepare_signed_tx",
        fake_prepare2,
    )

    async def fake_borrow(amount, token, inst, *, payer):
        return "sig"

    async def fake_repay(sig):
        return True

    monkeypatch.setattr(
        "solhunter_zero.agents.flashloan_sandwich.borrow_flash", fake_borrow
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.flashloan_sandwich.repay_flash", fake_repay
    )

    sent = []

    async def fake_submit(self, txs):
        sent.append(txs)

    monkeypatch.setattr(
        "solhunter_zero.agents.flashloan_sandwich.MEVExecutor.submit_bundle",
        fake_submit,
    )

    agent = FlashloanSandwichAgent(jito_rpc_url="ws://jito", jito_auth="A")
    token = asyncio.run(_run(agent))

    assert token == "tok"
    assert sent == [["TX_MSG_buy", "TX_MSG_sell"]]
