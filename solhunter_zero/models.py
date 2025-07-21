import os
from functools import lru_cache
from typing import Sequence

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
