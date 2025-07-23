import asyncio

from solhunter_zero.mev_executor import MEVExecutor


async def _run_exec(mev: MEVExecutor, txs):
    return await mev.submit_bundle(txs)


def test_mev_executor_submit(monkeypatch):
    calls = []

    async def fake_submit(tx, *, priority_rpc=None, priority_fee=None):
        calls.append((tx, priority_fee, priority_rpc))

    monkeypatch.setattr(
        "solhunter_zero.mev_executor.submit_raw_tx", fake_submit
    )
    monkeypatch.setattr(
        "solhunter_zero.mev_executor.snapshot", lambda tok: ({}, 5.0)
    )
    monkeypatch.setattr(
        "solhunter_zero.mev_executor.adjust_priority_fee", lambda rate: 7
    )

    mev = MEVExecutor("TOK", priority_rpc=["u"])
    asyncio.run(_run_exec(mev, ["A", "B"]))

    assert calls == [
        ("A", 7, ["u"]),
        ("B", 7, ["u"]),
    ]
