from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Position:
    token: str
    amount: float
    entry_price: float

@dataclass
class Portfolio:
    path: Optional[str] = "portfolio.json"
    balances: Dict[str, Position] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.load()

    # persistence helpers -------------------------------------------------
    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # pragma: no cover - invalid file
            return
        self.balances = {
            token: Position(token, info["amount"], info["entry_price"])
            for token, info in data.items()
        }

    def save(self) -> None:
        if not self.path:
            return
        data = {
            token: {"amount": pos.amount, "entry_price": pos.entry_price}
            for token, pos in self.balances.items()
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # position management -------------------------------------------------
    def add(self, token: str, amount: float, price: float) -> None:
        self.update(token, amount, price)

    def update(self, token: str, amount: float, price: float) -> None:
        pos = self.balances.get(token)
        if pos is None:
            self.balances[token] = Position(token, amount, price)
        else:
            total_cost = pos.amount * pos.entry_price + amount * price
            new_amount = pos.amount + amount
            if new_amount <= 0:
                self.balances.pop(token, None)
            else:
                pos.amount = new_amount
                if amount > 0:
                    pos.entry_price = total_cost / new_amount
        self.save()

    def remove(self, token: str) -> None:
        if token in self.balances:
            self.balances.pop(token)
            self.save()

    # analytics -----------------------------------------------------------
    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        pnl = 0.0
        for token, pos in self.balances.items():
            if token in prices:
                pnl += (prices[token] - pos.entry_price) * pos.amount
        return pnl
