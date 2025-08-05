import asyncio
import aiohttp

from solhunter_zero.mev_executor import MEVExecutor


async def _run_exec(mev: MEVExecutor, txs):
    return await mev.submit_bundle(txs)


def test_jito_auth_failure(monkeypatch):
    """Missing credentials should result in None signatures."""

    class FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {}

        def raise_for_status(self):
            raise aiohttp.ClientError()

    class FakeSession:
        def post(self, url, json=None, headers=None, timeout=10):
            return FakeResp()

    async def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(
        "solhunter_zero.mev_executor.get_session", fake_get_session
    )

    mev = MEVExecutor("TOK", jito_rpc_url="http://jito")
    sigs = asyncio.run(_run_exec(mev, ["A", "B"]))

    assert sigs == [None, None]


def test_jito_connection_failure(monkeypatch):
    """Connection errors should also return None signatures."""

    class FakeSession:
        def post(self, url, json=None, headers=None, timeout=10):
            raise aiohttp.ClientConnectionError()

    async def fake_get_session():
        return FakeSession()

    monkeypatch.setattr(
        "solhunter_zero.mev_executor.get_session", fake_get_session
    )

    mev = MEVExecutor("TOK", jito_rpc_url="http://jito", jito_auth="T")
    sigs = asyncio.run(_run_exec(mev, ["A"]))

    assert sigs == [None]
