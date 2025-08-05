from solhunter_zero import investor_demo
from solhunter_zero import simple_memory


def test_memory_fallback(tmp_path, monkeypatch):
    """Ensure investor_demo uses SimpleMemory when Memory fails to initialise."""

    def boom(*_a, **_k):
        raise ImportError("no sqlalchemy")

    # Force Memory instantiation to fail
    monkeypatch.setattr(investor_demo, "Memory", boom)

    called: dict[str, object] = {}

    async def fake_log_trade(self, **kwargs):
        called["trade"] = kwargs
        return 1

    async def fake_list_trades(self, token: str, **kwargs):  # pragma: no cover - simple stub
        return [called.get("trade", {})]

    monkeypatch.setattr(simple_memory.SimpleMemory, "log_trade", fake_log_trade)
    monkeypatch.setattr(simple_memory.SimpleMemory, "list_trades", fake_list_trades)

    investor_demo.main(["--reports", str(tmp_path)])

    # SimpleMemory.log_trade should have been invoked
    assert "trade" in called
