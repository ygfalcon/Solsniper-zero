from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Default base URL for the metrics API. Can be overridden by the
# ``METRICS_BASE_URL`` environment variable or via a configuration file that
# sets this variable before the module is imported.
DEFAULT_METRICS_BASE_URL = "https://api.example.com"

@dataclass
class SimulationResult:
    success_prob: float
    expected_roi: float
    volume: float = 0.0
    liquidity: float = 0.0


def fetch_token_metrics(token: str) -> dict:
    """Fetch historical return metrics for ``token``.

    The function tries to retrieve data from a remote API. When the call fails
    for any reason, default values are returned.  The metrics include the mean
    daily return and the standard deviation of daily returns.
    """

    base_url = os.getenv("METRICS_BASE_URL", DEFAULT_METRICS_BASE_URL)
    url = f"{base_url.rstrip('/')}/token/{token}/metrics"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {
            "mean": float(data.get("mean_return", 0.0)),
            "volatility": float(data.get("volatility", 0.02)),
            "volume": float(data.get("volume_24h", 0.0)),
            "liquidity": float(data.get("liquidity", 0.0)),
        }
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch metrics for %s: %s", token, exc)
        return {"mean": 0.0, "volatility": 0.02, "volume": 0.0, "liquidity": 0.0}


def run_simulations(
    token: str,
    count: int = 1000,
    days: int = 30,
    *,
    min_volume: float = 0.0,
) -> List[SimulationResult]:
    """Run Monte Carlo simulations for a given token."""

    metrics = fetch_token_metrics(token)
    if metrics.get("volume", 0.0) < min_volume:
        return []

    mu = metrics["mean"]
    sigma = metrics["volatility"]
    volume = metrics.get("volume", 0.0)
    liquidity = metrics.get("liquidity", 0.0)

    results: List[SimulationResult] = []
    for _ in range(count):
        daily_returns = np.random.normal(mu, sigma, days)
        roi = float(np.prod(1 + daily_returns) - 1)
        success_prob = float(np.mean(daily_returns > 0))
        results.append(SimulationResult(success_prob, roi, volume, liquidity))

    return results
