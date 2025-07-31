from __future__ import annotations

import json
import os
from typing import Iterable, Dict, Any

from .agents import BaseAgent

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

from .models import load_compiled_model


class SupervisorAgent(BaseAgent):
    """Load a policy checkpoint to select or weight strategies."""

    name = "supervisor"

    def __init__(self, checkpoint: str = "supervisor.json", device: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch and hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
        self.policy: Dict[str, float] = {}
        self.model = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint or not os.path.exists(self.checkpoint):
            return
        if self.checkpoint.endswith(".json"):
            try:
                with open(self.checkpoint, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self.policy = {str(k): float(v) for k, v in data.items()}
            except Exception:  # pragma: no cover - invalid file
                self.policy = {}
        else:
            try:
                self.model = load_compiled_model(self.checkpoint, self.device)
            except Exception:  # pragma: no cover - load failure
                self.model = None

    def predict_weights(
        self,
        agent_names: Iterable[str],
        token: str | None = None,
        portfolio: Any | None = None,
    ) -> Dict[str, float]:
        names = list(agent_names)
        if self.model is not None and torch is not None:
            try:
                with torch.no_grad():  # pragma: no cover - simple inference
                    x = torch.zeros((1, len(names)), dtype=torch.float32)
                    out = self.model(x)
                vals = out.squeeze().tolist()
                if not isinstance(vals, list):
                    vals = [float(vals)] * len(names)
                return {n: float(vals[i]) for i, n in enumerate(names)}
            except Exception:  # pragma: no cover - inference failure
                pass
        return {n: self.policy.get(n, 1.0) for n in names}

    async def propose_trade(
        self,
        token: str,
        portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> list[Dict[str, Any]]:
        return []
