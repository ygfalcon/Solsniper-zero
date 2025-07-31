import asyncio
import time
import types
import numpy as np
import pytest

from solhunter_zero.offline_data import OfflineData
import solhunter_zero.rl_training as rl_training
from scripts import build_mmap_dataset


@pytest.mark.asyncio
async def test_mmap_startup_benchmark(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db_url)
    await data.log_snapshot("tok", 1.0, 1.0, total_depth=1.0, imbalance=0.0)
    await data.log_trade("tok", "buy", 1.0, 1.0)

    mmap_dir = tmp_path / "datasets"
    mmap_dir.mkdir()
    mmap_path = mmap_dir / "offline_data.npz"

    def fake_main(args=None):
        time.sleep(0.01)
        np.savez_compressed(
            mmap_path,
            snapshots=np.zeros(0, dtype=[('token','U32'),('price','f4'),('depth','f4'),('total_depth','f4'),('slippage','f4'),('volume','f4'),('imbalance','f4'),('tx_rate','f4'),('whale_share','f4'),('spread','f4'),('sentiment','f4'),('timestamp','f8')]),
            trades=np.zeros(0, dtype=[('token','U32'),('side','U8'),('price','f4'),('amount','f4'),('timestamp','f8')]),
        )
        return 0

    monkeypatch.setattr(build_mmap_dataset, "main", fake_main)
    monkeypatch.setattr(rl_training, "_TradeDataset", lambda *a, **k: types.SimpleNamespace(__len__=lambda self: 0, __getitem__=lambda self, i: (0,0,0)))
    monkeypatch.setattr(rl_training.torch, "save", lambda *a, **k: None)
    monkeypatch.setattr(rl_training.pl.Trainer, "fit", lambda *a, **k: None)

    start = time.perf_counter()
    rl_training.fit([], [], model_path=tmp_path/'m.pt', db_url=db_url, mmap_path=str(mmap_path))
    first = time.perf_counter() - start

    start = time.perf_counter()
    rl_training.fit([], [], model_path=tmp_path/'m.pt', db_url=db_url, mmap_path=str(mmap_path))
    second = time.perf_counter() - start

    assert second <= first
