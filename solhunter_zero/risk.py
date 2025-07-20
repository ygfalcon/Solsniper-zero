from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    """Manage dynamic risk parameters."""

    risk_tolerance: float = 0.1
    max_allocation: float = 0.2
    max_risk_per_token: float = 0.1
    max_drawdown: float = 1.0
    volatility_factor: float = 1.0
    risk_multiplier: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskManager":
        """Create ``RiskManager`` from configuration dictionary."""
        return cls(
            risk_tolerance=float(cfg.get("risk_tolerance", 0.1)),
            max_allocation=float(cfg.get("max_allocation", 0.2)),
            max_risk_per_token=float(cfg.get("max_risk_per_token", 0.1)),
            max_drawdown=float(cfg.get("max_drawdown", 1.0)),
            volatility_factor=float(cfg.get("volatility_factor", 1.0)),
            risk_multiplier=float(cfg.get("risk_multiplier", 1.0)),
        )

    def adjusted(self, drawdown: float = 0.0, volatility: float = 0.0) -> "RiskManager":
        """Return a new ``RiskManager`` with parameters adjusted for conditions."""
        factor = max(0.0, 1 - drawdown / self.max_drawdown)
        scale = factor / (1 + volatility * self.volatility_factor)
        scale *= self.risk_multiplier
        return RiskManager(
            risk_tolerance=self.risk_tolerance * scale,
            max_allocation=self.max_allocation * scale,
            max_risk_per_token=self.max_risk_per_token * self.risk_multiplier,
            max_drawdown=self.max_drawdown,
            volatility_factor=self.volatility_factor,
            risk_multiplier=self.risk_multiplier,
        )
