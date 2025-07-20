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
    high_price: float = 0.0

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
            token: Position(
                token,
                info["amount"],
                info["entry_price"],
                info.get("high_price", info["entry_price"]),
            )
            for token, info in data.items()
        }

    def save(self) -> None:
        if not self.path:
            return
        data = {
            token: {
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "high_price": pos.high_price,
            }
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
            self.balances[token] = Position(token, amount, price, price)
        else:
            total_cost = pos.amount * pos.entry_price + amount * price
            new_amount = pos.amount + amount
            if new_amount <= 0:
                self.balances.pop(token, None)
            else:
                pos.amount = new_amount
                if amount > 0:
                    pos.entry_price = total_cost / new_amount
                if price > pos.high_price:
                    pos.high_price = price
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

    def update_highs(self, prices: Dict[str, float]) -> None:
        """Update high water marks for held tokens."""
        for token, pos in self.balances.items():
            price = prices.get(token)
            if price is not None and price > pos.high_price:
                pos.high_price = price
        self.save()

    def trailing_stop_triggered(self, token: str, price: float, trailing: float) -> bool:
        """Return ``True`` if ``price`` hits the trailing stop for ``token``."""
        pos = self.balances.get(token)
        if pos is None:
            return False
        if price > pos.high_price:
            pos.high_price = price
            self.save()
            return False
        return price <= pos.high_price * (1 - trailing)

    def percent_allocated(self, token: str, prices: Dict[str, float] | None = None) -> float:
        """Return the fraction of portfolio value allocated to ``token``."""
        prices = prices or {}
        total = self.total_value(prices)
        if total <= 0:
            return 0.0
        pos = self.balances.get(token)
        if pos is None:
            return 0.0
        price = prices.get(token, pos.entry_price)
        return (pos.amount * price) / total


def calculate_order_size(
    balance: float,
    expected_roi: float,
    volatility: float = 0.0,
    drawdown: float = 0.0,
    *,
    risk_tolerance: float = 0.1,
    max_allocation: float = 0.2,
    max_risk_per_token: float = 0.1,
    max_drawdown: float = 1.0,
    volatility_factor: float = 1.0,
    gas_cost: float | None = None,
    current_allocation: float = 0.0,
    min_portfolio_value: float = 0.0,
) -> float:
    """Return trade size based on ``balance`` and expected ROI.

    The position size grows with the expected return but is limited by the
    ``risk_tolerance`` fraction of the balance.  ``current_allocation``
    represents the portion of the portfolio already allocated to the token and
    ensures the total allocation never exceeds ``max_allocation``.  Negative or
    zero expected returns yield a size of ``0.0``.

    ``min_portfolio_value`` sets a lower bound on the balance used for sizing,
    preventing portfolios that dip below this value from generating trades that
    cannot cover network fees.
    """

    if balance <= 0 or expected_roi <= 0:
        return 0.0

    if drawdown >= max_drawdown:
        return 0.0

    adj_risk = risk_tolerance * (1 - drawdown / max_drawdown) / (
        1 + volatility * volatility_factor
    )
    fraction = expected_roi * adj_risk
    remaining = max_allocation - current_allocation
    max_fraction = min(max_risk_per_token, remaining)
    fraction = min(fraction, max_fraction)
    if fraction <= 0:
        return 0.0
    effective_balance = max(balance, min_portfolio_value)
    size = min(effective_balance * fraction, balance)
    if gas_cost is None:
        try:
            from .gas import get_current_fee
            gas_cost = get_current_fee()
        except Exception:
            gas_cost = 0.0
    if size <= gas_cost:
        return 0.0
    return size - gas_cost
