import argparse
import asyncio
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from solhunter_zero.offline_data import OfflineData
from solhunter_zero import models
from solhunter_zero.regime import detect_regime


def build_loader(
    data: OfflineData,
    seq_len: int,
    batch_size: int,
    regime: str | None = None,
):
    """Return DataLoader and feature dimension."""
    snaps = data.list_snapshots()
    if len(snaps) <= seq_len:
        return None, 0
    prices = [float(s.price) for s in snaps]
    X_full, y_full = models.make_snapshot_training_data(snaps, seq_len=seq_len)
    regimes = [
        detect_regime(prices[i : i + seq_len + 1]) for i in range(len(prices) - seq_len)
    ]
    if regime is not None:
        idx = [i for i, r in enumerate(regimes) if r == regime]
        if not idx:
            return None, 0
        X_full = X_full[idx]
        y_full = y_full[idx]
    dataset = TensorDataset(X_full, y_full)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, X_full.size(-1)


def load_or_create_model(path: Path, device: torch.device, input_dim: int) -> torch.nn.Module:
    model = models.get_model(str(path))
    if model is None or getattr(model, "input_dim", input_dim) != input_dim:
        model = models.DeepTransformerModel(input_dim)
    model.to(device)
    return model


async def train_loop(
    db_url: str,
    model_path: Path,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    interval: float,
    regime: str | None,
) -> None:
    data = OfflineData(db_url)
    loader, input_dim = build_loader(data, seq_len, batch_size, regime)
    if loader is None:
        return
    model = load_or_create_model(model_path, device, input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    while True:
        loader, _ = build_loader(data, seq_len, batch_size, regime)
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
    p.add_argument("--regime", default=None)
    args = p.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    await train_loop(
        args.db,
        Path(args.model),
        device,
        args.seq_len,
        args.batch_size,
        args.interval,
        args.regime,
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
