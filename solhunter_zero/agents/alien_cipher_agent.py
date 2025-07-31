from __future__ import annotations

import hashlib
import json
from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class AlienCipherAgent(BaseAgent):
    """Logistic-map trading decisions from predefined coefficients."""

    name = "alien_cipher"

    def __init__(
        self,
        threshold: float = 0.5,
        amount: float = 1.0,
        dataset_path: str = "datasets/alien_cipher.json",
    ) -> None:
        self.threshold = float(threshold)
        self.amount = float(amount)
        self.dataset_path = dataset_path

    # ------------------------------------------------------------------
    def _load_coeffs(self, token: str) -> tuple[float, int] | None:
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        info = data.get(token)
        if not isinstance(info, dict):
            return None
        try:
            r = float(info.get("r", 0.0))
            iterations = int(info.get("iterations", 0))
        except Exception:
            return None
        return r, iterations

    # ------------------------------------------------------------------
    def _initial_value(self, token: str) -> float:
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        return digest / float(1 << 256)

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        coeffs = self._load_coeffs(token)
        if coeffs is None:
            return []
        r, iterations = coeffs
        x = self._initial_value(token)
        for _ in range(max(0, iterations)):
            x = r * x * (1.0 - x)

        if x >= self.threshold:
            return [{"token": token, "side": "buy", "amount": self.amount, "price": 0.0}]
        if x <= 1.0 - self.threshold and token in portfolio.balances:
            pos = portfolio.balances[token]
            return [{"token": token, "side": "sell", "amount": pos.amount, "price": 0.0}]
        return []
