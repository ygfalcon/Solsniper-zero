import argparse
import asyncio
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from solhunter_zero.offline_data import OfflineData
from solhunter_zero import models


def build_loader(data: OfflineData, seq_len: int, batch_size: int) -> DataLoader | None:
    snaps = data.list_snapshots()
    if len(snaps) <= seq_len:
        return None
    X, y = models.make_snapshot_training_data(snaps, seq_len=seq_len)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_or_create_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = models.get_model(str(path))
    if model is None:
        model = models.DeepTransformerModel(6)
    model.to(device)
    return model


async def train_loop(db_url: str, model_path: Path, device: torch.device, seq_len: int,
                     batch_size: int, interval: float) -> None:
    data = OfflineData(db_url)
    model = load_or_create_model(model_path, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    while True:
        loader = build_loader(data, seq_len, batch_size)
        if loader is not None:
            model.train()
            for X_b, y_b in loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)
                opt.zero_grad()
                pred = model(X_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                opt.step()
            models.save_model(model.to("cpu"), str(model_path))
            model.to(device)
        await asyncio.sleep(interval)


async def main() -> None:
    p = argparse.ArgumentParser(description="Continuous transformer training")
    p.add_argument("--db", default="sqlite:///offline_data.db")
    p.add_argument("--model", default="transformer_model.pt")
    p.add_argument("--interval", type=float, default=3600.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    await train_loop(args.db, Path(args.model), device, args.seq_len, args.batch_size, args.interval)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
