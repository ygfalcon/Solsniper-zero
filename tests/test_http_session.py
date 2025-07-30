import asyncio
import aiohttp
import pytest

from solhunter_zero.http import get_session, close_session, session_context


def test_get_and_close_session_no_warning(recwarn):
    async def runner():
        session = await get_session()
        assert isinstance(session, aiohttp.ClientSession)
        await close_session()
    asyncio.run(runner())
    assert not recwarn


def test_session_context_manager(recwarn):
    async def runner():
        async with session_context() as session:
            assert isinstance(session, aiohttp.ClientSession)
    asyncio.run(runner())
    assert not recwarn
