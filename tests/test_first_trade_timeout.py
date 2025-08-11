import os
import asyncio
import logging
import pytest

os.environ.setdefault("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
os.environ.setdefault("DEX_BASE_URL", "https://quote-api.jup.ag")
os.environ.setdefault("AGENTS", "[\"dummy\"]")
_cfg_path = os.path.join(os.path.dirname(__file__), "tmp_config.toml")
with open(_cfg_path, "w", encoding="utf-8") as _f:
    _f.write("solana_rpc_url='https://api.mainnet-beta.solana.com'\n")
    _f.write("dex_base_url='https://quote-api.jup.ag'\n")
    _f.write("agents=['dummy']\n")
    _f.write("agent_weights={dummy=1.0}\n")
os.environ["SOLHUNTER_CONFIG"] = _cfg_path

# Import the trading loop module directly to access first trade helpers
import solhunter_zero.loop as main  # noqa: E402


def test_first_trade_detected(caplog):
    caplog.set_level(logging.ERROR)
    main._first_trade_recorded = False
    main._first_trade_event = asyncio.Event()

    async def trigger():
        await asyncio.sleep(0.01)
        main._first_trade_event.set()
        main._first_trade_recorded = True

    async def runner():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(trigger())
            tg.create_task(main._check_first_trade(0.1, retry=False))

    asyncio.run(runner())
    assert "First trade not recorded" not in caplog.text


def test_first_trade_timeout(caplog):
    caplog.set_level(logging.ERROR)
    main._first_trade_recorded = False
    main._first_trade_event = asyncio.Event()

    async def trigger():
        await asyncio.sleep(0.1)

    async def runner():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(trigger())
            tg.create_task(main._check_first_trade(0.05, retry=True))

    with pytest.raises(main.FirstTradeTimeoutError):
        try:
            asyncio.run(runner())
        except* main.FirstTradeTimeoutError as eg:
            # Re-raise the captured FirstTradeTimeoutError to satisfy pytest
            raise eg.exceptions[0]
    assert "First trade not recorded" in caplog.text
