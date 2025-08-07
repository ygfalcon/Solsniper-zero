import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_STAGING_TRADE") != "1",
    reason="RUN_STAGING_TRADE not enabled",
)


def test_main_loop_executes_trade(monkeypatch):
    rpc = os.getenv("SOLANA_RPC_URL")
    keypair = os.getenv("KEYPAIR_PATH")
    token = os.getenv("STAGING_TOKEN")
    if not rpc or not keypair or not token:
        pytest.skip("SOLANA_RPC_URL, KEYPAIR_PATH, STAGING_TOKEN required")

    from solhunter_zero import main as main_module
    from solhunter_zero import wallet, routeffi, depth_client

    monkeypatch.setenv("SOLANA_RPC_URL", rpc)
    monkeypatch.setenv("KEYPAIR_PATH", keypair)
    monkeypatch.setenv("USE_DEPTH_STREAM", "1")
    monkeypatch.setenv("AGENTS", "")

    wallet_loaded = {"flag": False}

    def wallet_wrapper(path: str):
        wallet_loaded["flag"] = True
        return wallet.load_keypair(path)

    monkeypatch.setattr(wallet, "load_keypair", wallet_wrapper)

    if not routeffi.is_routeffi_available():
        pytest.skip("route_ffi library not available")

    route_called = {"flag": False}
    real_best_route = routeffi.best_route

    def route_wrapper(*args, **kwargs):
        route_called["flag"] = True
        return real_best_route(*args, **kwargs)

    monkeypatch.setattr(routeffi, "best_route", route_wrapper)

    depth_called = {"flag": False}
    real_snapshot = depth_client.snapshot

    def depth_wrapper(tok: str):
        depth_called["flag"] = True
        return real_snapshot(tok)

    monkeypatch.setattr(depth_client, "snapshot", depth_wrapper)

    async def fake_discover(self, **_):
        return [token]

    monkeypatch.setattr(main_module.DiscoveryAgent, "discover_tokens", fake_discover)

    class DummyStrategyManager:
        def __init__(self, *args, **kwargs):
            pass

        async def evaluate(self, token: str, portfolio):
            await depth_client.best_route(token, 1.0)
            depth_client.snapshot(token)
            return [{"token": token, "side": "buy", "amount": 0.000001, "price": 0.0}]

        def list_missing(self):
            return []

    monkeypatch.setattr(main_module, "StrategyManager", DummyStrategyManager)

    trades: list[dict] = []

    async def fake_log_trade(self, **kw):
        trades.append(kw)

    monkeypatch.setattr(main_module.Memory, "log_trade", fake_log_trade)

    main_module.main(iterations=1, loop_delay=0, memory_path="sqlite:///:memory:")

    assert wallet_loaded["flag"], "wallet was not loaded"
    assert route_called["flag"], "route_ffi was not invoked"
    assert depth_called["flag"], "depth_service was not queried"
    assert trades, "no trade was logged"
