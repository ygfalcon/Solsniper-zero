"""Simple in-memory portfolio management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass
class Position:
    token: str
    amount: float
    entry_price: float

    def value(self, current_price: float) -> float:
        return self.amount * current_price


@dataclass
class Portfolio:
    balances: Dict[str, Position] = field(default_factory=dict)

    def add(self, token: str, amount: float, price: float) -> None:
        if token in self.balances:
            pos = self.balances[token]
            total_amt = pos.amount + amount
            pos.entry_price = (pos.entry_price * pos.amount + price * amount) / total_amt
            pos.amount = total_amt
        else:
            self.balances[token] = Position(token, amount, price)

    def remove(self, token: str) -> None:
        self.balances.pop(token, None)

    def iterate(self) -> Iterable[Position]:
        return self.balances.values()
