import os
from functools import lru_cache
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


class PriceModel(nn.Module):
    """Simple LSTM based predictor."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1]
        return self.fc(out).squeeze(-1)

    def predict(self, seq: Sequence[Sequence[float]]) -> float:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            pred = self(t)
            return float(pred.item())


def save_model(model: PriceModel, path: str) -> None:
    """Save ``model`` to ``path`` using a portable format."""
    torch.save(
        {
            "cfg": {
                "input_dim": model.lstm.input_size,
                "hidden_dim": model.lstm.hidden_size,
                "num_layers": model.lstm.num_layers,
            },
            "state": model.state_dict(),
        },
        path,
    )


def load_model(path: str) -> PriceModel:
    """Load a :class:`PriceModel` from ``path``."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, PriceModel):
        obj.eval()
        return obj
    if isinstance(obj, dict) and "state" in obj:
        cfg = obj.get("cfg", {})
        model = PriceModel(**cfg)
        model.load_state_dict(obj["state"])
        model.eval()
        return model
    raise TypeError("Invalid model file")


@lru_cache(None)
def _cached_load(path: str) -> PriceModel:
    return load_model(path)


def get_model(path: str | None) -> PriceModel | None:
    """Return model from path if it exists, otherwise ``None``."""
    if not path or not os.path.exists(path):
        return None
    try:
        return _cached_load(path)
    except Exception:
        return None


def make_training_data(
    prices: Iterable[float],
    liquidity: Iterable[float],
    depth: Iterable[float],
    tx_counts: Iterable[float] | None = None,
    seq_len: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create model inputs from historical metrics."""

    p = torch.tensor(list(prices), dtype=torch.float32)
    l = torch.tensor(list(liquidity), dtype=torch.float32)
    d = torch.tensor(list(depth), dtype=torch.float32)
    if tx_counts is None:
        t = torch.zeros_like(p)
    else:
        t = torch.tensor(list(tx_counts), dtype=torch.float32)

    n = len(p) - seq_len
    if n <= 0:
        raise ValueError("not enough history for seq_len")

    seqs = []
    targets = []
    for i in range(n):
        seq = torch.stack(
            [p[i : i + seq_len], l[i : i + seq_len], d[i : i + seq_len], t[i : i + seq_len]],
            dim=1,
        )
        seqs.append(seq)
        p0 = p[i + seq_len - 1]
        p1 = p[i + seq_len]
        targets.append((p1 - p0) / p0)

    X = torch.stack(seqs)
    y = torch.tensor(targets, dtype=torch.float32)
    return X, y


def train_price_model(
    prices: Iterable[float],
    liquidity: Iterable[float],
    depth: Iterable[float],
    tx_counts: Iterable[float] | None = None,
    *,
    seq_len: int = 30,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
) -> PriceModel:
    """Train a :class:`PriceModel` on historical data."""

    X, y = make_training_data(prices, liquidity, depth, tx_counts, seq_len)
    model = PriceModel(4, hidden_dim=hidden_dim, num_layers=num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    model.eval()
    return model
