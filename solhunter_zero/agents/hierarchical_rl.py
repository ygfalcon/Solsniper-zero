from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Dict, Any, List

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - lightweight fallback
    import types

    class _Dummy(nn.Module if 'nn' in globals() else object):
        pass

    torch = types.ModuleType("torch")
    torch.Tensor = list
    torch.tensor = lambda *a, **k: [0.0 for _ in range(len(a[0]) if a else 1)]
    torch.softmax = lambda x, dim=0: [1.0 / len(x) for _ in x]
    torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self,*exc: None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    nn = types.ModuleType("nn")
    class _Layer:
        def __call__(self, x):
            return x
        def to(self, *a, **k):
            return self
        def eval(self):
            pass
        def load_state_dict(self, *a, **k):
            pass
    nn.Module = _Layer
    nn.Linear = lambda *a, **k: _Layer()
    nn.ReLU = lambda *a, **k: _Layer()
    nn.Sequential = lambda *a, **k: _Layer()

from . import BaseAgent
from ..rl_daemon import portfolio_state
from ..portfolio import Portfolio


class HierarchicalRLAgent(BaseAgent):
    """Predict agent weighting coefficients from current state."""

    name = "hierarchical_rl"

    def __init__(
        self,
        agent_names: Iterable[str],
        *,
        model_path: str | os.PathLike = "hier_rl.pt",
        device: str | None = None,
    ) -> None:
        self.agent_names = list(agent_names)
        self.model_path = Path(model_path)
        if device is None:
            if getattr(torch.cuda, "is_available", lambda: False)():
                device = "cuda"
            elif getattr(torch.backends.mps, "is_available", lambda: False)():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.agent_names)),
        )
        if self.model_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            data = torch.load(self.model_path, map_location=self.device)
            if isinstance(data, dict) and "state" in data:
                self.model.load_state_dict(data.get("state", {}))
                self.agent_names = list(data.get("agents", self.agent_names))
            else:
                self.model.load_state_dict(data)
        except Exception:  # pragma: no cover - corrupt file
            pass

    def reload_weights(self) -> None:
        if self.model_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def compute_weights(
        self,
        portfolio: Portfolio,
        token: str,
        price: float,
        *,
        depth: float = 0.0,
        tx_rate: float = 0.0,
    ) -> Dict[str, float]:
        state = portfolio_state(portfolio, token, price, depth=depth, tx_rate=tx_rate)
        tensor = torch.tensor([state], dtype=getattr(torch, "float32", None), device=self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)[0]
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
        probs = torch.softmax(logits, dim=0)
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        return {name: float(probs[i]) for i, name in enumerate(self.agent_names)}

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        return []
