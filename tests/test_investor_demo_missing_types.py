import pytest
from pathlib import Path

from solhunter_zero import investor_demo

pytestmark = pytest.mark.timeout(30)


def test_investor_demo_missing_types(tmp_path, monkeypatch):
    class DummyMem:
        def __init__(self, *a, **k):
            pass

        def log_var(self, value: float) -> None:
            pass

        async def close(self) -> None:  # pragma: no cover - simple stub
            pass

    def fake_hedge(weights, corrs):
        return weights

    monkeypatch.setattr(investor_demo, "Memory", DummyMem)
    monkeypatch.setattr(investor_demo, "hedge_allocation", fake_hedge)

    async def noop() -> None:
        return None

    monkeypatch.setattr(investor_demo, "_demo_flash_loan", noop)

    data = Path(__file__).resolve().parent / "data" / "prices_short.json"

    with pytest.raises(RuntimeError) as exc_info:
        investor_demo.main([
            "--data",
            str(data),
            "--reports",
            str(tmp_path),
        ])

    msg = str(exc_info.value)
    assert "Demo did not exercise trade types" in msg
    assert "flash_loan" in msg
