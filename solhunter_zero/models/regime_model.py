from __future__ import annotations

from typing import Sequence, Iterable, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REGIME_LABELS = ["bull", "bear", "sideways"]


class RegimeModel(nn.Module):
    """LSTM based classifier for market regimes."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_layers: int = 2, seq_len: int = 30) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(REGIME_LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1]
        return self.fc(out)

    def predict(self, seq: Sequence[Sequence[float]]) -> str:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            logits = self(t)
            idx = int(torch.argmax(logits, dim=1).item())
        return REGIME_LABELS[idx]


def make_training_data(snaps: Sequence[Any], seq_len: int = 30, threshold: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct dataset from offline snapshots."""

    prices = torch.tensor([float(s.price) for s in snaps], dtype=torch.float32)
    liq = torch.tensor([float(getattr(s, "depth", 0.0)) for s in snaps], dtype=torch.float32)
    sent = torch.tensor([float(getattr(s, "sentiment", 0.0)) for s in snaps], dtype=torch.float32)

    n = len(prices) - seq_len
    if n <= 0:
        raise ValueError("not enough history for seq_len")

    seqs = []
    labels = []
    for i in range(n):
        seq = torch.stack([
            prices[i:i+seq_len],
            liq[i:i+seq_len],
            sent[i:i+seq_len],
        ], dim=1)
        seqs.append(seq)
        p0 = prices[i + seq_len - 1]
        p1 = prices[i + seq_len]
        change = (p1 - p0) / p0
        if change > threshold:
            labels.append(0)
        elif change < -threshold:
            labels.append(1)
        else:
            labels.append(2)

    X = torch.stack(seqs)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def train_regime_model(
    snaps: Sequence[Any],
    *,
    seq_len: int = 30,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
    threshold: float = 0.02,
) -> RegimeModel:
    """Train :class:`RegimeModel` using offline snapshots."""

    X, y = make_training_data(snaps, seq_len=seq_len, threshold=threshold)
    model = RegimeModel(X.size(-1), hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def save_regime_model(model: RegimeModel, path: str) -> None:
    cfg = {
        "cls": "RegimeModel",
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "seq_len": model.seq_len,
    }
    torch.save({"cfg": cfg, "state": model.state_dict()}, path)


def load_regime_model(path: str) -> RegimeModel:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, RegimeModel):
        obj.eval()
        return obj
    if isinstance(obj, dict) and "state" in obj:
        cfg = obj.get("cfg", {})
        cls_name = cfg.pop("cls", "RegimeModel")
        if cls_name != "RegimeModel":
            raise TypeError("Invalid regime model type")
        model = RegimeModel(**cfg)
        model.load_state_dict(obj["state"])
        model.eval()
        return model
    raise TypeError("Invalid model file")

