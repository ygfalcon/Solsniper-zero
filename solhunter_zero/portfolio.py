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
    max_value: float = 0.0

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

    def position_roi(self, token: str, price: float) -> float:
        """Return the return-on-investment for ``token`` at ``price``.

        The ROI is expressed as a fraction of the entry price.  If the token
        is not held in the portfolio or the entry price is zero, ``0.0`` is
        returned.
        """
        pos = self.balances.get(token)
        if pos is None or pos.entry_price == 0:
            return 0.0
        return (price - pos.entry_price) / pos.entry_price

    def total_value(self, prices: Dict[str, float]) -> float:
        """Return portfolio value using ``prices`` or entry prices."""
        value = 0.0
        for token, pos in self.balances.items():
            price = prices.get(token, pos.entry_price)
            value += pos.amount * price
        return value

    def update_drawdown(self, prices: Dict[str, float]) -> None:
        """Update maximum portfolio value for drawdown calculations."""
        value = self.total_value(prices)
        if value > self.max_value:
            self.max_value = value

    def current_drawdown(self, prices: Dict[str, float]) -> float:
        """Return current drawdown fraction based on ``prices``."""
        value = self.total_value(prices)
        if self.max_value == 0:
            self.max_value = value
            return 0.0
        return (self.max_value - value) / self.max_value


def calculate_order_size(
    balance: float,
    expected_roi: float,
    volatility: float = 0.0,
    drawdown: float = 0.0,
    *,
    risk_tolerance: float = 0.1,
    max_allocation: float = 0.2,
    max_risk_per_token: float = 0.1,
) -> float:
    """Return trade size based on ``balance`` and expected ROI.

    The position size grows with the expected return but is limited by the
    ``risk_tolerance`` and ``max_allocation`` fractions of the balance.
    Negative or zero expected returns yield a size of ``0.0``.
    """

    if balance <= 0 or expected_roi <= 0:
        return 0.0

    adj_risk = risk_tolerance * (1 - drawdown) / (1 + volatility)
    fraction = expected_roi * adj_risk
    fraction = min(fraction, max_allocation, max_risk_per_token)
    if fraction <= 0:
        return 0.0
    return balance * fraction
