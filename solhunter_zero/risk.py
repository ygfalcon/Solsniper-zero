from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .memory import Memory


def value_at_risk(
    prices: Sequence[float], confidence: float = 0.95, memory: Memory | None = None
) -> float:
    """Return historical Value-at-Risk of ``prices``.

    ``prices`` should be ordered oldest to newest.  The VaR is returned as a
    positive fraction representing the maximum expected loss at the given
    confidence level.
    """

    if len(prices) < 2:
        var = 0.0
    else:
        returns = [prices[i + 1] / prices[i] - 1 for i in range(len(prices) - 1)]
        returns.sort()
        idx = int((1 - confidence) * len(returns))
        idx = max(0, min(idx, len(returns) - 1))
        var = -returns[idx]
        if var < 0:
            var = 0.0

    if memory is not None:
        try:  # pragma: no cover - logging failures are non-critical
            memory.log_var(var)
        except Exception:
            pass

    return var


@dataclass
class RiskManager:
    """Manage dynamic risk parameters."""

    risk_tolerance: float = 0.1
    max_allocation: float = 0.2
    max_risk_per_token: float = 0.1
    max_drawdown: float = 1.0
    volatility_factor: float = 1.0
    risk_multiplier: float = 1.0
    min_portfolio_value: float = 20.0
    funding_rate_factor: float = 1.0
    sentiment_factor: float = 1.0
    token_age_factor: float = 30.0

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
            min_portfolio_value=float(cfg.get("min_portfolio_value", 20.0)),
            funding_rate_factor=float(cfg.get("funding_rate_factor", 1.0)),
            sentiment_factor=float(cfg.get("sentiment_factor", 1.0)),
            token_age_factor=float(cfg.get("token_age_factor", 30.0)),
        )

    def adjusted(
        self,
        drawdown: float = 0.0,
        volatility: float = 0.0,
        *,
        volume_spike: float = 1.0,
        depth_change: float = 0.0,
        whale_activity: float = 0.0,
        tx_rate: float = 0.0,
        portfolio_value: float | None = None,
        funding_rate: float = 0.0,
        sentiment: float = 0.0,
        token_age: float | None = None,
        prices: Sequence[float] | None = None,
        var_threshold: float | None = None,
        var_confidence: float = 0.95,
        memory: Memory | None = None,
    ) -> "RiskManager":
        """Return a new ``RiskManager`` adjusted using recent market metrics.

        Parameters
        ----------
        drawdown:
            Current portfolio drawdown as a fraction of ``max_drawdown``.
        volatility:
            Recent price volatility.
        volume_spike:
            Multiplicative factor representing sudden volume increase.
        depth_change:
            Change in order book depth from :mod:`onchain_metrics`.
        whale_activity:
            Fraction of liquidity controlled by large wallets.
        tx_rate:
            Mempool transaction rate from :mod:`onchain_metrics`.
        portfolio_value:
            Current portfolio USD value.  When below ``min_portfolio_value`` the
            scaling factor is reduced further.
        """

        factor = max(0.0, 1 - drawdown / self.max_drawdown)
        scale = factor / (1 + volatility * self.volatility_factor)
        if volume_spike > 1:
            scale *= min(volume_spike, 2.0)
        if tx_rate > 1:
            scale *= min(tx_rate, 2.0)
        scale /= 1 + abs(depth_change)
        scale /= 1 + whale_activity
        scale *= self.risk_multiplier

        if funding_rate:
            if funding_rate > 0:
                scale *= 1 + funding_rate * self.funding_rate_factor
            else:
                scale /= 1 + abs(funding_rate) * self.funding_rate_factor

        if sentiment:
            scale *= 1 + sentiment * self.sentiment_factor

        if token_age is not None and self.token_age_factor > 0:
            age_scale = min(1.0, token_age / self.token_age_factor)
            scale *= age_scale

        if portfolio_value is not None and portfolio_value < self.min_portfolio_value:
            pv_scale = max(0.0, portfolio_value / self.min_portfolio_value)
            scale *= pv_scale

        if prices is not None and var_threshold is not None:
            var = value_at_risk(prices, var_confidence, memory=memory)
            if var > var_threshold and var > 0:
                scale *= var_threshold / var
        return RiskManager(
            risk_tolerance=self.risk_tolerance * scale,
            max_allocation=self.max_allocation * scale,
            max_risk_per_token=self.max_risk_per_token * scale,
            max_drawdown=self.max_drawdown,
            volatility_factor=self.volatility_factor,
            risk_multiplier=self.risk_multiplier,
            min_portfolio_value=self.min_portfolio_value,
            funding_rate_factor=self.funding_rate_factor,
            sentiment_factor=self.sentiment_factor,
            token_age_factor=self.token_age_factor,
        )
