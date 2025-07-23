from __future__ import annotations

import math
from typing import List, Dict, Any

from . import BaseAgent
from ..portfolio import Portfolio


class CardinalAgent(BaseAgent):
    """Adaptive ordinal strategy cycling based on regret entropy."""

    name = "abraxas"

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self.regret_lattice: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.modal_weights: Dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0}
        self.order: List[int] = [1, 2, 3]
        self.last_ordinal: int | None = None
        self.last_outcome: bool | None = None

        self.regret_entropy_curve: List[float] = []
        self.modal_ordinal_delta: float = 0.0
        self.inversion_gate_triggered: bool = False
        self.symbolic_execution_stack: List[int] = []

    # ------------------------------------------------------------------
    def _update_from_outcome(self) -> None:
        if self.last_ordinal is None or self.last_outcome is None:
            return
        regret = self.regret_lattice[self.last_ordinal]
        if self.last_outcome:
            regret = max(regret - 1, 0)
        else:
            regret += 1
        self.regret_lattice[self.last_ordinal] = regret
        entropy = math.log(regret + 1)
        self.regret_entropy_curve.append(entropy)

        for i in (1, 2, 3):
            self.modal_weights[i] = 1.0 / (1.0 + self.regret_lattice[i])
        self.modal_ordinal_delta = self.modal_weights[1] - self.modal_weights[2]

        inverted = self.order == [3, 1, 2]
        if entropy > self.threshold and not inverted:
            self.order = [3, 1, 2]
            self.inversion_gate_triggered = True
        elif entropy <= self.threshold and inverted:
            self.order = [1, 2, 3]
            self.inversion_gate_triggered = False
        else:
            self.inversion_gate_triggered = entropy > self.threshold
        self.last_outcome = None

    # ------------------------------------------------------------------
    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        self._update_from_outcome()
        stack = list(self.order)
        ordinal = stack[0]
        self.order = stack[1:] + stack[:1]
        self.symbolic_execution_stack = stack
        self.last_ordinal = ordinal

        side_map = {1: "buy", 2: "sell", 3: "buy"}
        amount_map = {1: 1.0, 2: 1.0, 3: 2.0}
        side = side_map[ordinal]
        if self.inversion_gate_triggered and ordinal == 2:
            side = "buy" if side == "sell" else "sell"
        action = {
            "token": token,
            "side": side,
            "amount": amount_map[ordinal],
            "price": 0.0,
        }
        return [action]
