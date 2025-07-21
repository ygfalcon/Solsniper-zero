from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

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


def conditional_value_at_risk(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Return Conditional Value-at-Risk (expected shortfall) of ``returns``."""

    if len(returns) == 0:
        return 0.0
    arr = np.sort(np.asarray(returns, dtype=float))
    cutoff = int((1 - confidence) * len(arr))
    cutoff = max(1, cutoff)
    tail = arr[:cutoff]
    cvar = -float(tail.mean())
    return max(cvar, 0.0)


def covariance_matrix(prices: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Return covariance matrix of token returns."""

    series = []
    for seq in prices.values():
        arr = np.asarray(seq, dtype=float)
        if len(arr) < 2:
            continue
        rets = arr[1:] / arr[:-1] - 1
        series.append(rets)
    if not series:
        return np.empty((0, 0))
    min_len = min(len(s) for s in series)
    mat = np.vstack([s[:min_len] for s in series])
    return np.cov(mat)


def portfolio_cvar(
    prices: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> float:
    """Return portfolio CVaR for ``prices`` and ``weights``."""

    series = []
    w_list = []
    for tok, w in weights.items():
        seq = prices.get(tok)
        if seq is None or len(seq) < 2:
            continue
        arr = np.asarray(seq, dtype=float)
        rets = arr[1:] / arr[:-1] - 1
        series.append(rets)
        w_list.append(w)
    if not series:
        return 0.0
    min_len = min(len(s) for s in series)
    mat = np.vstack([s[:min_len] for s in series]).T
    w = np.asarray(w_list, dtype=float)
    port_rets = mat @ w
    return conditional_value_at_risk(port_rets, confidence)


def portfolio_variance(cov: np.ndarray, weights: Sequence[float]) -> float:
    """Return portfolio variance given covariance ``cov`` and ``weights``."""

    if cov.size == 0:
        return 0.0
    w = np.asarray(list(weights), dtype=float)
    if cov.shape[0] != w.size:
        return 0.0
    return float(w @ cov @ w.T)


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
        covariance: float | None = None,
        covar_threshold: float | None = None,
        portfolio_cvar: float | None = None,
        cvar_threshold: float | None = None,
        leverage: float | None = None,
        correlation: float | None = None,
        regime: str | None = None,
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
        covariance:
            Portfolio return covariance measure.  When exceeding ``covar_threshold``
            risk is scaled down.
        portfolio_cvar:
            Conditional VaR of the portfolio returns.
        leverage:
            Target leverage factor for dynamic scaling.
        correlation:
            Average correlation across held assets used for hedging.
        regime:
            Optional market regime label (``"bull"``, ``"bear"`` or ``"sideways"``)
            that influences scaling.
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

        if covariance is not None and covar_threshold is not None and covariance > covar_threshold:
            scale *= covar_threshold / covariance

        if portfolio_cvar is not None and cvar_threshold is not None and portfolio_cvar > cvar_threshold:
            scale *= cvar_threshold / portfolio_cvar

        if correlation is not None:
            corr = max(-1.0, min(1.0, correlation))
            scale *= max(0.0, 1 - corr)

        if leverage is not None and leverage > 0:
            scale *= leverage

        if regime:
            reg = regime.lower()
            if reg == "bull":
                scale *= 1.2
            elif reg == "bear":
                scale *= 0.8
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
