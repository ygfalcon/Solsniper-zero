from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Position:
    token: str
    amount: float
    entry_price: float

@dataclass
class Portfolio:
    balances: Dict[str, Position] = field(default_factory=dict)

    def add(self, token: str, amount: float, price: float) -> None:
        self.balances[token] = Position(token, amount, price)

    def remove(self, token: str) -> None:
        self.balances.pop(token, None)
