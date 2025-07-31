from solhunter_zero.offline_data import OfflineData
import pytest


@pytest.mark.asyncio
async def test_offline_data_roundtrip(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    await data.log_snapshot("tok", 1.0, 2.0, total_depth=3.0, imbalance=0.5, slippage=0.0, volume=0.0)
    snaps = await data.list_snapshots("tok")
    assert snaps and snaps[0].token == "tok"
    assert snaps[0].tx_rate == 0.0
    assert snaps[0].whale_share == 0.0
    assert snaps[0].spread == 0.0
    assert snaps[0].sentiment == 0.0
    await data.log_trade("tok", "buy", 1.0, 2.0)
    trades = await data.list_trades("tok")
    assert trades and trades[0].side == "buy"


@pytest.mark.asyncio
async def test_async_queue_commit(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession
    import psutil
    import asyncio

    commits = 0
    orig_commit = AsyncSession.commit

    async def counting_commit(self, *a, **k):
        nonlocal commits
        commits += 1
        return await orig_commit(self, *a, **k)

    monkeypatch.setattr(AsyncSession, "commit", counting_commit)

    db = f"sqlite:///{tmp_path/'direct.db'}"
    data = OfflineData(db)
    for _ in range(5):
        await data.log_snapshot("tok", 1.0, 1.0, imbalance=0.0)
    assert commits == 5

    commits = 0
    data_q = OfflineData(f"sqlite:///{tmp_path/'queued.db'}")
    data_q.start_writer(batch_size=5, interval=0.01)
    proc = psutil.Process()
    start = proc.cpu_times().user
    for _ in range(5):
        await data_q.log_snapshot("tok", 1.0, 1.0, imbalance=0.0)
    await asyncio.sleep(0.05)
    await data_q.close()
    queued_cpu = proc.cpu_times().user - start

    assert commits <= 1
    assert queued_cpu >= 0
